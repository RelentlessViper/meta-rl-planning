# %%
import os
from dataclasses import dataclass
import time

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium import register
import numpy as np
from typing import Optional, Union
import matplotlib.pyplot as plt
import random
import draccus

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, TensorDataset

from src.environments.dark_room_wrappers import RL2DarkRoom
#from src.environments.gym_record_video import RecordVideo

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
class ProbePlotConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = False
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "rl2-darkroom-meta"
    capture_video: bool = False
    save_best_model: bool = False

    # Algorithm specific arguments
    env_id: str = "Dark-Room-5x5-v0"
    num_trials: int = 3
    hidden_size: int = 512
    num_layers: int = 1
    num_episodes: int = 1
    model_checkpoint_path: str = None
    probe_checkpoint_path: str = None
    
    def __post_init__(self):
        if self.model_checkpoint_path is None:
            raise ValueError(f"`model_checkpoint_path` must be filled, got: {self.model_checkpoint_path}")
        if self.probe_checkpoint_path is None:
            raise ValueError(f"`probe_checkpoint_path` must be filled, got: {self.probe_checkpoint_path}")
        self.run_name = f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"

class ProbeVisualizationWrapper(gym.Wrapper):
    def __init__(self, env, probe_model, device="cpu"):
        super().__init__(env)
        self.probe_model = probe_model
        self.device = device
        self.last_probe_grid = None
        self.grid_size = self.env.unwrapped.size # 5 for 5x5
        
        # Mapping action indices to strings for rendering
        self.action_names = {
            0: "✕",
            1: "↑",
            2: "→",
            3: "↓",
            4: "←",
            5: None, # "Never Visited"
        }

    def reset(self, **kwargs):
        # 1. Clear or initialize the probe grid to the "Never Visited" label (index 5)
        # This ensures the very first frame captured by RecordVideo has the 600x600 shape
        self.last_probe_grid = np.full((self.grid_size, self.grid_size), 5, dtype=np.int64)
        
        # 2. Proceed with the standard reset
        return self.env.reset(**kwargs)

    def update_probe_prediction(self, hidden_state):
        """
        Updates the internal grid of predicted actions based on the GRU hidden state.
        """
        with torch.no_grad():
            # Ensure hidden_state is a tensor: (1, 512)
            if not isinstance(hidden_state, torch.Tensor):
                hidden_state = torch.tensor(hidden_state, dtype=torch.float32)
            
            hidden_state = hidden_state.to(self.device)
            # Logits shape: (1, 25, 6)
            logits = self.probe_model(hidden_state)
            # Sample or take Argmax to get labels: (25,)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1).squeeze(0).cpu().numpy()
            
            # Reshape 25 vector to 5x5 grid
            self.last_probe_grid = predictions.reshape(self.grid_size, self.grid_size)

    def render(self):
        """
        Custom render that takes the base RGB array and overlays text labels.
        """
        img = self.env.render() # Get (5, 5, 3) or higher res RGB array
        
        #if self.last_probe_grid is None:
        #    return img

        # Use Matplotlib to draw the text labels over the RGB array
        fig, ax = plt.subplots(figsize=(6, 6), dpi=600)
        ax.imshow(img, extent=[0, self.grid_size, self.grid_size, 0])
        
        # Overlay probe predictions
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                action_idx = self.last_probe_grid[i, j]
                label = self.action_names[action_idx]
                
                # Center text in cell
                if label is not None:
                    ax.text(j + 0.5, i + 0.5, label, 
                            color='black', ha='center', va='center', 
                            fontsize=8, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        ax.set_xticks(np.arange(self.grid_size + 1))
        ax.set_yticks(np.arange(self.grid_size + 1))
        ax.grid(True)
        
        # Convert plt figure back to RGB array
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        data = np.asarray(rgba_buffer)
        
        # Convert RGBA to RGB (dropping the alpha channel)
        data = data[:, :, :3]
        
        plt.close(fig)
        return data

def make_env(
    env_id,
    run_name,
    probe_model,
    device,
    num_trials
):
    env = gym.make(env_id)
    env = RL2DarkRoom(env, trials_per_episode=num_trials)
    #env = RL2ProbeWrapper(env, trials_per_episode=num_trials)
    env = ProbeVisualizationWrapper(env, probe_model, device)
    env = RecordVideo(
        env,
        f"videos/{run_name}",
        name_prefix="probing",
        episode_trigger=lambda episode_id: True,
        disable_logger=True,
    )
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
    
class GridActionProbe(nn.Module):
    def __init__(self, hidden_dim=512, grid_size=5, num_actions=6):
        super().__init__()
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size
        self.num_actions = num_actions
        self.linear = nn.Linear(hidden_dim, self.num_cells * num_actions)

    def forward(self, hidden_states):
        B = hidden_states.size(0)
        logits = self.linear(hidden_states) # [B,150]
        return logits.view(B, self.num_cells, self.num_actions)

@draccus.wrap()
def render_probe_output(args: ProbePlotConfig):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    probes = []
    for idx, probe_weights in enumerate(torch.load(args.probe_checkpoint_path).values()):
        probe = GridActionProbe()
        probe.load_state_dict(probe_weights)
        probe.eval()
        probes.append(probe)

    env = make_env(
        args.env_id,
        args.run_name,
        probes[0],
        device,
        args.num_trials,
    )

    agent = Agent(env, args.hidden_size, args.num_layers)
    agent.load_state_dict(torch.load(args.model_checkpoint_path, weights_only=True)["model_state_dict"])
    agent.eval()

    for episode in range(args.num_episodes):
        next_obs, info = env.reset()
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(1).to(device) # We have only one env
        next_lstm_state = torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device)

        for step in range(env.spec.max_episode_steps * args.num_trials):
            action, _, _, _, next_lstm_state = agent.get_action_and_value(
                next_obs,
                next_lstm_state,
                next_done
            )
            env.env.update_probe_prediction(next_lstm_state)
            next_obs, reward, terminated, truncatated, info = env.step(action.item())
            next_obs = torch.tensor(next_obs).to(device).float()
            next_done = torch.tensor(np.logical_or(terminated, truncatated)).to(device).float()

            if terminated or truncatated:
                break

if __name__ == "__main__":
    render_probe_output()