import os
import random
from collections import deque
import time
from copy import deepcopy

import draccus
from dataclasses import dataclass
from tqdm import trange, tqdm
import toymeta
import gymnasium as gym
from gymnasium.envs import register
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import TensorDataset

from box_world_env import RL2BoxWorld, RevealChestContentsWrapper
from conv_gru import ConvGRU

@dataclass
class DatasetCollectionConfig:
    dataset_name: str = None
    seed: int = 1
    num_episodes: int = 10_000
    num_trials: int = 3
    save_path: str = None
    model_checkpoint_path: str = None
    hidden_size: int = 32
    num_layers: int = 1
    cuda: bool = True

    # Box World arguments
    field_size: int = 7
    goal_length: int = 3
    num_distractor: int = 0
    distractor_length: int = 0
    step_cost: float = 0.0
    reward_gem: float = 1.0
    reward_key: float = 0.0
    reward_distractor: float = 0.0
    collect_key: bool = True
    max_episode_timesteps: int = 64
    
    def __post_init__(self):
        self.env_id = f"Box-World-{self.field_size}x{self.field_size}-{self.goal_length}-{self.num_distractor}-v0"
        
        if self.dataset_name is None:
            self.dataset_name = f"{self.env_id}__{int(time.time())}"
        
        if not self.save_path:
            self.save_path = f"datasets/"
        
        register(
            id=self.env_id,
            entry_point="box_world_env.box_world_env:BoxWorld",
            max_episode_steps=int(self.max_episode_timesteps),
            kwargs=dict(
                n=self.field_size,
                goal_length=self.goal_length,
                num_distractor=self.num_distractor,
                distractor_length=self.distractor_length,
                max_steps=int(self.max_episode_timesteps),
                collect_key=self.collect_key,
                step_cost = self.step_cost,
                reward_gem = self.reward_gem,
                reward_key = self.reward_key,
                reward_distractor = self.reward_distractor,
            ),
        )
        self.dataset_len = gym.make(self.env_id).spec.max_episode_steps * self.num_trials * self.num_episodes

def make_env(env_id, num_trials):
    env = gym.make(env_id, collect_key=False)
    env = RevealChestContentsWrapper(env)
    env = RL2BoxWorld(env, trials_per_episode=num_trials)
    return env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ConvGRUAgent(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_dim,
        num_actions,
        num_layers=1,
    ):
        super().__init__()
        self.height, self.width = input_shape[1:]
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_layers = num_layers

        self.in_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0], out_channels=hidden_dim, kernel_size=1
            ),
            nn.ReLU(),
        )
        self.rnn = ConvGRU(
            input_shape=(hidden_dim, self.height, self.width),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = layer_init(
            nn.Linear(self.hidden_dim * self.height * self.width, num_actions),
            std=0.01,
        )
        self.critic = layer_init(
            nn.Linear(self.hidden_dim * self.height * self.width, 1),
            std=1.0,
        )

    def get_state(self, x, gru_state, done):
        hidden = self.in_proj(x)

        # RNN logic
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape(
            (
                -1,
                batch_size,
                self.hidden_dim,
                self.height,
                self.width,
            )
        )
        done = done.reshape((-1, batch_size))

        new_hidden = []
        for h, d in zip(hidden, done):
            h, gru_state = self.rnn(
                h.unsqueeze(0),
                (1.0 - d).view(1, -1, 1, 1, 1) * gru_state,
            )
            new_hidden += [h.flatten(-3, -1)]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, gru_state

    def get_value(self, x, gru_state, done):
        hidden, _ = self.get_state(x, gru_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, gru_state, done, action=None, argmax=False):
        hidden, gru_state = self.get_state(x, gru_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if argmax:
            action = logits.argmax(dim=-1)
            return (
                action,
                probs.log_prob(action),
                probs.entropy(),
                self.critic(hidden),
                gru_state,
            )
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
            gru_state,
        )

def onehot_to_xy(onehot_tensor, grid_size=5):
    idx = torch.argmax(onehot_tensor).item()
    
    x = idx % grid_size
    y = idx // grid_size
    
    return torch.tensor([y, x])

@draccus.wrap()
def collect_trajectories(args: DatasetCollectionConfig):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(args.env_id, args.num_trials)

    agent = ConvGRUAgent(
        input_shape=(env.observation_space.shape[0], args.field_size + 2, args.field_size + 2),
        num_layers=args.num_layers,
        hidden_dim=args.hidden_size,
        num_actions=env.action_space.n,
    ).to(device)
    agent = torch.compile(agent)

    if args.model_checkpoint_path is not None:
        agent.load_state_dict(torch.load(args.model_checkpoint_path, weights_only=True, map_location=device)["model_state_dict"])
    agent.eval()
    actions = []
    player_pos = []
    hidden_states = []
    observations = []
    trial_idxs = []

    trial_data = []

    with torch.no_grad():
        for episode in trange(args.num_episodes, desc="Collection"):
            next_obs, info = env.reset()
            next_obs = torch.from_numpy(next_obs).to(device)
            next_done = torch.zeros(1).to(device)
            next_hidden_state = agent.rnn.init_hidden(1) # Since we are using only one env

            for step in range(env.spec.max_episode_steps * env.trials_per_episode):
                action, _, _, _, next_hidden_state = agent.get_action_and_value(
                    next_obs,
                    next_hidden_state,
                    next_done,
                    argmax=True,
                )
                current_position = env.unwrapped.player_position.copy()
                player_pos += [current_position]
                hidden_states += [next_hidden_state[-1].squeeze(0).clone()] # If we have multiple layers, use hidden state of the last layer and squeeze the batch/env dim
                actions += [action.reshape(())]
                observations += [next_obs]
                trial_idxs += [nn.functional.one_hot(torch.tensor(env.trial_counter), num_classes=env.trials_per_episode)]

                next_obs, reward, terminated, truncated, info = env.step(action.item())
                next_done = int(terminated or truncated)
                next_obs, next_done = torch.from_numpy(next_obs).to(device), torch.Tensor([next_done]).to(device)

                if info["trial_done"]:
                    trial_data += [
                        {
                            "hidden_state": hidden_states,
                            "action": actions,
                            "observation": observations,
                            "player_pos": player_pos,
                            "trial_idx": trial_idxs,
                        }
                    ]
                    hidden_states, actions, observations, player_pos, trial_idxs = [], [], [], [], []

                if terminated or truncated:
                    break

    print("Data collection is finished.")
    # Reverse: define a grid with all first future actions for each hidden state
    grid_size = env.unwrapped.n ** 2
    # for idx, trial in enumerate(trial_data):
    features = []
    targets = []
    for idx, cur_trial in enumerate(tqdm(trial_data, desc="Postprocessing")):
        grid_state = torch.ones(
            (
                env.observation_space.shape[-2],
                env.observation_space.shape[-1],
            )
        ).to(device) * 5 # 5 denotes a cell that was never visited
        grid_states = deque()

        for idx, (hidden_state, action, observation, player_position, trial_idx) in enumerate(
            zip(
                reversed(cur_trial["hidden_state"]),
                reversed(cur_trial["action"]),
                reversed(cur_trial["observation"]),
                reversed(cur_trial["player_pos"]),
                reversed(cur_trial["trial_idx"]),
            )
        ):
            current_pos = player_position.tolist()
            grid_state[current_pos[0]][current_pos[1]] = deepcopy(action)
            for i in range(1, grid_state.size(0) - 1): # Exclude walls
                for j in range(1, grid_state.size(1) - 1):
                    features.append(hidden_state[:, i, j].clone())
                    targets.append(deepcopy(grid_state[i][j]))

    features_tensor = torch.stack(features)
    targets_tensor = torch.stack(targets)

    features_len_before = features_tensor.size(0)
    # Leave only unique elements
    unique_mask = []
    seen = set()
    for i in range(features_tensor.size(0)):
        key = torch.cat(
            [
                features_tensor[i].view(-1),
                targets_tensor[i].view(-1),
            ]
        ).cpu().numpy().tobytes()
        if key not in seen:
            seen.add(key)
            unique_mask.append(i)
    unique_indices = torch.tensor(unique_mask, dtype=torch.long)

    features_tensor = features_tensor[unique_indices]
    targets_tensor = targets_tensor[unique_indices]
    features_len_after = features_tensor.size(0)
    print(f"Size with duplicates: {features_len_before}")
    print(f"Size without duplicates: {features_len_after}")

    dataset = TensorDataset(
        features_tensor,
        targets_tensor,
    )

    os.makedirs(args.save_path, exist_ok=True)

    torch.save(dataset, f'{args.save_path}/{args.dataset_name}.pt')
    print(f"Saved dataset for all trials:")

    print(f"  - Location: {args.save_path}/{args.dataset_name}.pt")
    print(f"  - Samples: {len(dataset)}")
    print(f"  - Shapes: {[t.shape for t in dataset.tensors]}")


if __name__ == "__main__":
    collect_trajectories()