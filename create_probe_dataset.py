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
from torch.utils.data import Dataset, TensorDataset
from src.environments.dark_room_wrappers import RL2DarkRoom

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
    def thunk():
        env = gym.make(env_id)
        env = RL2DarkRoom(env, trials_per_episode=num_trials)
        #env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, hidden_size, num_layers):
        super().__init__()
        self.in_proj = nn.Sequential(
            layer_init(nn.Linear(np.prod(envs.single_observation_space.shape), hidden_size)),
            nn.ReLU(),
        )
        self.lstm = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(hidden_size, envs.single_action_space.n), std=0.01)
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

class TrialDataset(Dataset):
    def __init__(self, trial_data):
        self.trial_data = trial_data
        
    def __len__(self):
        return len(self.trial_data)
    
    def __getitem__(self, idx):
        item = self.trial_data[idx]
        return {
            'hidden_state': item['hidden_state'],
            'action': item['action'],
            'observation': item['observation'],
            'grid_state': item['grid_state']
        }

def onehot_to_xy(onehot_tensor, grid_size=5):
    idx = torch.argmax(onehot_tensor).item()
    
    x = idx % grid_size
    y = idx // grid_size
    
    return torch.tensor([y, x])

@draccus.wrap()
def collect_trajectories(args: DatasetCollectionConfig):
    args = DatasetCollectionConfig()
    args.model_checkpoint_path = "checkpoints/Dark-Room-5x5-v0__hidden-512_envs-128_3-trials__1__1771760503_best.pt"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = gym.vector.SyncVectorEnv([make_env(args.env_id, args.num_trials)])

    agent = Agent(env, args.hidden_size, args.num_layers)
    agent.load_state_dict(torch.load(args.model_checkpoint_path, weights_only=True)["model_state_dict"])
    agent.eval()

    next_obs, info = env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(env.num_envs).to(device)
    next_lstm_state = torch.zeros(agent.lstm.num_layers, env.num_envs, agent.lstm.hidden_size).to(device)

    actions = []
    hidden_states = []
    observations = []
    trial_data = [[] for _ in range(args.num_trials)]

    with torch.no_grad():
        for step in trange(args.dataset_len):
            action, _, _, _, next_lstm_state = agent.get_action_and_value(
                next_obs,
                next_lstm_state,
                next_done,
            )
            hidden_states += [next_lstm_state.reshape((-1,))]
            actions += [action.reshape(())]
            observations += [next_obs.reshape((-1,))]

            trial_counter = env.envs[0].trial_counter
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            next_done = np.logical_or(terminated, truncated)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if info["trial_done"][0]:
                observations[0][:25] = torch.Tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
                trial_data[trial_counter] += [
                    {
                        "hidden_state": hidden_states,
                        "action": actions,
                        "observation": observations,
                    }
                ]
                hidden_states, actions, observations = [], [], []

        print("Data collection is finished.")
        # Reverse: define a grid with all first future actions for each hidden state
        grid_size = env.envs[0].unwrapped.size ** 2
        for idx, trial in enumerate(trial_data):
            for cur_trial in trial:
                grid_state = torch.ones(
                    (
                        env.envs[0].unwrapped.size,
                        env.envs[0].unwrapped.size,
                    )
                ).to(device) * -1 # -1 denotes a cell that was never visited
                hidden_states = cur_trial["hidden_state"]
                actions = cur_trial["action"]
                observations = cur_trial["observation"]
                grid_states = deque()

                for idx, (hidden_state, action, observation) in enumerate(
                    zip(
                        reversed(hidden_states),
                        reversed(actions),
                        reversed(observations),
                    )
                ):
                    
                    current_pos = onehot_to_xy(observation[:grid_size], env.envs[0].unwrapped.size)
                    grid_state[current_pos[0]][current_pos[1]] = action
                    grid_states.appendleft(grid_state.clone())
                
                cur_trial["grid_state"] = list(grid_states)

    env.close()

    trial_datasets = []
        
    for trial_idx, trial in enumerate(trial_data):
        hidden_states = torch.cat([torch.stack(item['hidden_state']) for item in trial])
        actions = torch.cat([torch.stack(item['action']) for item in trial])
        observations = torch.cat([torch.stack(item['observation']) for item in trial])
        grid_states = torch.cat([torch.stack(item['grid_state']) for item in trial])
        
        dataset = TensorDataset(hidden_states, actions, observations, grid_states)
        trial_datasets.append(dataset)

    os.makedirs(args.save_path, exist_ok=True)
    for i, dataset in enumerate(trial_datasets):
        all_tensors = dataset.tensors

        torch.save(all_tensors, f'{args.save_path}/trial_{i}_dataset.pt')

        metadata = {
            'trial_idx': i,
            'num_samples': len(dataset),
            'tensor_shapes': [t.shape for t in all_tensors],
            'tensor_dtypes': [t.dtype for t in all_tensors]
        }
        torch.save(metadata, f'{args.save_path}/trial_{i}_metadata.pt')

        print(f"Saved trial {i} dataset:")
        print(f"  - Location: {args.save_path}/trial_{i}_dataset.pt")
        print(f"  - Samples: {len(dataset)}")
        print(f"  - Shapes: {[t.shape for t in all_tensors]}")


if __name__ == "__main__":
    collect_trajectories()