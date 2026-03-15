import os
import random
from collections import deque

import draccus
from dataclasses import dataclass
from tqdm import trange
import toymeta
import gymnasium as gym
from gymnasium.envs import register
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import TensorDataset
from dark_room_wrappers import RL2DarkRoom

register(
    id="Dark-Room-5x5-v0",
    entry_point="toymeta.dark_room:DarkRoom",
    max_episode_steps=15,
    kwargs={
        "size": 5,
        "random_start": False,
        "terminate_on_goal": False,
    },
)

@dataclass
class DatasetCollectionConfig:
    dataset_name: str = "probe-dataset-5x5"
    seed: int = 1
    num_episodes: int = 5_000
    num_trials: int = 3
    probe_type: str = "one_for_all_trials"
    """The type of probe we want to train. Can be either 'one_for_all_trials' or 'one_for_each_trial'"""
    env_id: str = "Dark-Room-5x5-v0"
    save_path: str = None
    model_checkpoint_path: str = None
    hidden_size: int = 512
    num_layers: int = 1
    cuda: bool = True
    
    def __post_init__(self):
        self.dataset_len = gym.make(self.env_id).spec.max_episode_steps * self.num_trials * self.num_episodes # 5x5 setting: 3*15*5000 = 225000
        if not self.save_path:
            self.save_path = f"datasets/{self.dataset_name}"

def make_env(env_id, num_trials):
    env = gym.make(env_id)
    env = RL2DarkRoom(env, trials_per_episode=num_trials)
    return env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, hidden_size, num_layers):
        super().__init__()
        self.in_proj = nn.Sequential(
            layer_init(nn.Linear(np.prod(envs.observation_space.shape), hidden_size)),
            nn.ReLU(),
        )
        self.lstm = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(hidden_size, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1)

    def get_state(self, x, lstm_state, done):
        hidden = self.in_proj(x)

        # LSTM logic
        batch_size = lstm_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (1.0 - d).view(1, -1, 1) * lstm_state,
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_state(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_state(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state

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
    agent = Agent(env, args.hidden_size, args.num_layers)
    if args.model_checkpoint_path is not None:
        agent.load_state_dict(torch.load(args.model_checkpoint_path, weights_only=True)["model_state_dict"])
    agent.eval()
    actions = []
    hidden_states = []
    observations = []
    trial_idxs = []

    if args.probe_type == "one_for_all_trials":
        trial_data = [[]]
    elif args.probe_type == "one_for_each_trial":
        trial_data = [[] for _ in range(args.num_trials)]
    else:
        raise ValueError("`probe_type` can either be 'one_for_all_trials' or `one_for_each_trial`")

    with torch.no_grad():
        for episode in trange(args.num_episodes):
            next_obs, info = env.reset()
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(1).to(device)
            next_lstm_state = torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device)

            for step in range(env.spec.max_episode_steps * env.trials_per_episode):
                action, _, _, _, next_lstm_state = agent.get_action_and_value(
                    next_obs,
                    next_lstm_state,
                    next_done,
                )
                hidden_states += [next_lstm_state.reshape((-1,))]
                actions += [action.reshape(())]
                observations += [next_obs.reshape((-1,))]
                trial_idxs += [nn.functional.one_hot(torch.tensor(env.trial_counter), num_classes=env.trials_per_episode)]

                trial_counter = env.trial_counter if args.probe_type == "one_for_each_trial" else 0
                next_obs, reward, terminated, truncated, info = env.step(action.item())
                next_done = int(terminated or truncated)
                next_obs, next_done = torch.Tensor(next_obs.reshape((1, -1))).to(device), torch.Tensor([next_done]).to(device)

                if info["trial_done"]:
                    goal_pos = env.unwrapped.goal_pos[0] * env.unwrapped.size + env.unwrapped.goal_pos[1]
                    trial_data[trial_counter] += [
                        {
                            "hidden_state": hidden_states,
                            "action": actions,
                            "observation": observations,
                            "goal_pos": goal_pos,
                            "trial_idx": trial_idxs,
                        }
                    ]
                    hidden_states, actions, observations, trial_idxs = [], [], [], []

                if terminated or truncated:
                    break

    print("Data collection is finished.")

    # Reverse: define a grid with all first future actions for each hidden state
    grid_size = env.unwrapped.size ** 2
    for idx, trial in enumerate(trial_data):
        for cur_trial in trial:
            grid_state = torch.ones(
                (
                    env.unwrapped.size,
                    env.unwrapped.size,
                )
            ).to(device) * -1 # -1 denotes a cell that was never visited
            grid_states = deque()

            for idx, (hidden_state, action, observation, trial_idx) in enumerate(
                zip(
                    reversed(cur_trial["hidden_state"]),
                    reversed(cur_trial["action"]),
                    reversed(cur_trial["observation"]),
                    reversed(cur_trial["trial_idx"]),
                )
            ):
                current_pos = onehot_to_xy(observation[:grid_size], env.unwrapped.size)
                grid_state[current_pos[0]][current_pos[1]] = action
                grid_states.appendleft(grid_state.clone())
            
            cur_trial["grid_state"] = list(grid_states)

    trial_datasets = []

    for trial_idx, trial in enumerate(trial_data):
        trial_len = len(trial[0]["action"])
        hidden_states = torch.cat([torch.stack(item["hidden_state"]) for item in trial])
        actions = torch.cat([torch.stack(item["action"]) for item in trial])
        observations = torch.cat([torch.stack(item["observation"]) for item in trial])
        grid_states = torch.cat([torch.stack(item["grid_state"]) for item in trial])
        goal_pos = torch.cat([torch.tensor(item["goal_pos"]).reshape((1, -1)).expand(trial_len, -1) for item in trial]).reshape(-1)
        trial_idxs = torch.cat([torch.stack(item["trial_idx"]) for item in trial])
        
        # Leave only unique elements
        unique_mask = []
        seen = set()
        for i in range(hidden_states.shape[0]):
            # Create a hashable representation
            key = torch.cat([
                hidden_states[i].view(-1),
                #actions[i].view(-1),
                #observations[i].view(-1),
                grid_states[i].view(-1),
                trial_idxs[i].view(-1),
            ]).cpu().numpy().tobytes()

            if key not in seen:
                seen.add(key)
                unique_mask.append(i)
        
        unique_indices = torch.tensor(unique_mask, dtype=torch.long)

        hidden_states = hidden_states[unique_indices]
        actions = actions[unique_indices]
        observations = observations[unique_indices]
        grid_states = grid_states[unique_indices]
        goal_pos = goal_pos[unique_indices]
        trial_idxs = trial_idxs[unique_indices]

        dataset = TensorDataset(
            hidden_states,
            actions,
            observations,
            grid_states,
            goal_pos,
            trial_idxs,
        )
        trial_datasets.append(dataset)

    env.close()

    os.makedirs(args.save_path, exist_ok=True)
    for i, dataset in enumerate(trial_datasets):
        all_tensors = dataset.tensors
        metadata = {
            'trial_idx': i,
            'num_samples': len(dataset),
            'tensor_shapes': [t.shape for t in all_tensors],
            'tensor_dtypes': [t.dtype for t in all_tensors]
        }

        if args.probe_type == "one_for_all_trials":
            torch.save(all_tensors, f'{args.save_path}/all_trials_dataset.pt')
            torch.save(metadata, f'{args.save_path}/all_trials_metadata.pt')
            print(f"Saved dataset for all trials:")
        else:
            torch.save(all_tensors, f'{args.save_path}/trial_{i}_dataset.pt')
            torch.save(metadata, f'{args.save_path}/trial_{i}_metadata.pt')
            print(f"Saved trial {i} dataset:")

        print(f"  - Location: {args.save_path}/trial_{i}_dataset.pt")
        print(f"  - Samples: {len(dataset)}")
        print(f"  - Shapes: {[t.shape for t in all_tensors]}")


if __name__ == "__main__":
    collect_trajectories()