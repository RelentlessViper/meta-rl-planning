import os
from dataclasses import dataclass
import time

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium import register
import numpy as np
import matplotlib.pyplot as plt
import random
import draccus

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from dark_room_wrappers import RL2DarkRoom

@dataclass
class ProbePlotConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = False
    cuda: bool = True
    wandb_project_name: str = "rl2-darkroom-meta"

    # Algorithm specific arguments
    env_id: str = None
    num_trials: int = 3
    hidden_size: int = 512
    num_layers: int = 1
    num_episodes: int = 2
    model_checkpoint_path: str = None
    action_probe_checkpoint_path: str = None
    goal_pos_probe_checkpoint_path: str = None

    # Dark Room arguments
    room_size: int = 5
    max_trial_timesteps: int = 15
    
    def __post_init__(self):
        if self.env_id is None:
            self.env_id = f"Dark-Room-{self.room_size}x{self.room_size}-v0"
            register(
                id=f"Dark-Room-{self.room_size}x{self.room_size}-v0",
                entry_point="toymeta.dark_room:DarkRoom",
                max_episode_steps=self.max_trial_timesteps,
                kwargs={
                    "size": self.room_size,
                    "random_start": False,
                    "terminate_on_goal": False,
                },
            )
        self.run_name = f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"

class ProbeVisualizationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        action_probe_model,
        goal_pos_probe_model,
        hidden_dim=515,
        trials_per_episode=3,
        device="cpu"
    ):
        super().__init__(env)

        self.action_probe_model = action_probe_model
        self.goal_pos_probe_model = goal_pos_probe_model
        self.device = device

        self.hidden_dim = hidden_dim
        self.trials_per_episode = trials_per_episode

        self.grid_size = self.env.unwrapped.size
        self.num_cells = self.grid_size ** 2

        self.current_trial = 0

        self.probe_grids = [
            np.full((2, self.grid_size, self.grid_size), -1, dtype=np.int64) # Action prediction + goal position prediction
            for _ in range(self.trials_per_episode)
        ]

        self.action_names = {
            0: "✕",
            1: "↑",
            2: "→",
            3: "↓",
            4: "←",
            5: None,
        }

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.current_trial = 0

        self.probe_grids = [
            [
                np.zeros((self.grid_size, self.grid_size)),  # action grid
                np.zeros((self.grid_size, self.grid_size))   # goal prob grid
            ]
            for _ in range(self.trials_per_episode)
        ]

        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if "current_trial" in info:
            self.current_trial = info["current_trial"] + 1

        return obs, reward, terminated, truncated, info

    def update_probe_prediction(self, hidden_state):

        with torch.no_grad():

            if not isinstance(hidden_state, torch.Tensor):
                hidden_state = torch.tensor(hidden_state, dtype=torch.float32)

            hidden_state = hidden_state.to(self.device)

            for trial in range(self.current_trial, self.trials_per_episode):

                # trial one-hot
                trial_one_hot = torch.zeros(1, self.trials_per_episode, device=self.device)
                trial_one_hot[0, trial] = 1

                probe_input = torch.cat(
                    [hidden_state.reshape((1, -1)), trial_one_hot],
                    dim=-1
                )  # [1, 512 + 3]

                action_logits = self.action_probe_model(probe_input)  # [1, 25, 6]
                goal_pos_logits = self.goal_pos_probe_model(probe_input) # [1, 25]

                action_probs = torch.softmax(action_logits, dim=-1)
                goal_pos_probs = torch.softmax(goal_pos_logits, dim=-1)

                action_preds = torch.argmax(action_probs, dim=-1)  # [1, 25]

                action_grid = action_preds.squeeze(0).cpu().numpy().reshape(
                    self.grid_size,
                    self.grid_size,
                )
                goal_pos_grid = torch.round(goal_pos_probs.squeeze(0), decimals=3).cpu().numpy().reshape(
                    self.grid_size,
                    self.grid_size,
                )

                self.probe_grids[trial][0] = action_grid.copy()
                self.probe_grids[trial][1] = goal_pos_grid.copy()

    def render(self):
        base_img = self.env.render()

        fig, axes = plt.subplots(
            2,
            self.trials_per_episode,
            figsize=(4 * self.trials_per_episode, 8),
            dpi=300
        )

        if self.trials_per_episode == 1:
            axes = np.array(axes).reshape(2, 1)

        for trial_idx, (ax_action, ax_goal_pos) in enumerate(zip(axes[0], axes[1])):

            grid = self.probe_grids[trial_idx]

            if trial_idx < self.current_trial:
                img = np.ones((self.grid_size, self.grid_size, 3))
                timeline_title = "(past)"
                hide_content = True
            elif trial_idx == self.current_trial:
                img = base_img
                timeline_title = "(current)"
                hide_content = False
            else:
                img = np.ones((self.grid_size, self.grid_size, 3))
                timeline_title = "(predicted)"
                hide_content = False
            
            for ax_type, ax in zip(["actions", "goal position"], [ax_action, ax_goal_pos]):
                ax.imshow(img, extent=[0, self.grid_size, self.grid_size, 0])
                ax.set_title(f"Trial {trial_idx+1} {ax_type} {timeline_title}")

            if not hide_content:
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        action_idx = int(grid[0][i, j])
                        if action_idx == -1:
                            action_idx = 5
                        goal_pos_prob = str(grid[1][i, j])

                        label = self.action_names[action_idx]
                        if action_idx != 5 and label is not None:
                            ax_action.text(
                                j + 0.5,
                                i + 0.5,
                                label,
                                ha="center",
                                va="center",
                                fontsize=10,
                                fontweight="bold",
                                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                            )
                        
                        ax_goal_pos.text(
                            j + 0.5,
                            i + 0.5,
                            goal_pos_prob,
                            ha="center",
                            va="center",
                            fontsize=5,
                            fontweight="bold",
                            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                        )
                    
            for ax in [ax_action, ax_goal_pos]:
                ax.set_xticks(np.arange(self.grid_size + 1))
                ax.set_yticks(np.arange(self.grid_size + 1))
                ax.grid(True)

        fig.tight_layout()
        fig.canvas.draw()

        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]

        plt.close(fig)

        return rgb

def make_env(
    env_id,
    run_name,
    action_probe_model,
    goal_pos_probe_model,
    device,
    num_trials
):
    env = gym.make(env_id)
    env = RL2DarkRoom(env, trials_per_episode=num_trials)
    #env = RL2ProbeWrapper(env, trials_per_episode=num_trials)
    env = ProbeVisualizationWrapper(env, action_probe_model, goal_pos_probe_model, device)
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
    def __init__(self, hidden_dim=515, num_actions=6, grid_size=5):
        super().__init__()
        self.num_actions = num_actions
        self.grid_size = grid_size
        self.linear = nn.Linear(hidden_dim, grid_size ** 2 * num_actions)

    def forward(self, hidden_states):
        x = self.linear(hidden_states) # [B,5x5x6]
        return x.reshape((-1, self.grid_size ** 2, self.num_actions)) # [B,25,6]
    
class GoalPosProbe(nn.Module):
    def __init__(self, hidden_dim=515, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        self.linear = nn.Linear(hidden_dim, grid_size ** 2)

    def forward(self, hidden_states):
        return self.linear(hidden_states) # [B,25]

@draccus.wrap()
def render_probe_output(args: ProbePlotConfig):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    action_probes = []
    for idx, probe_weights in enumerate(torch.load(args.action_probe_checkpoint_path).values()):
        action_probe = GridActionProbe(grid_size=args.room_size)
        if args.action_probe_checkpoint_path is not None:
            action_probe.load_state_dict(probe_weights)
        action_probe.eval()
        action_probes.append(action_probe)
    
    goal_pos_probes = []
    for idx, probe_weights in enumerate(torch.load(args.goal_pos_probe_checkpoint_path).values()):
        goal_pos_probe = GoalPosProbe(grid_size=args.room_size)
        if args.goal_pos_probe_checkpoint_path is not None:
            goal_pos_probe.load_state_dict(probe_weights)
        goal_pos_probe.eval()
        goal_pos_probes.append(goal_pos_probe)

    env = make_env(
        args.env_id,
        args.run_name,
        action_probes[0],
        goal_pos_probes[0],
        device,
        args.num_trials,
    )

    agent = Agent(env, args.hidden_size, args.num_layers)
    if args.model_checkpoint_path is not None:
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