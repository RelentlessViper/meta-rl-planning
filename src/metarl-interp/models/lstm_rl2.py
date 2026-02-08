import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .preprocessing import preprocess_input

class RL2LSTMPolicy(nn.Module):
    """
    LSTM Policy for RL^2 with hidden state reset mechanics with 2 output heads: Action head & Value head.
    """
    def __init__(
        self,
        obs_size,
        action_size,
        hidden_size,
        num_layers,
    ):
        super().__init__()
        self.in_size = obs_size + action_size + 2 # We will concatenate observations with previous actions, rewards, and "done"s
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc = nn.Linear(
            in_features=self.in_size,
            out_features=hidden_size,
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.action_head = nn.Linear(
            in_features=hidden_size,
            out_features=action_size,
        )
        self.value_head = nn.Linear(
            in_features=hidden_size,
            out_features=1,
        )
        self._reset_hidden_state()

    def _get_device(self):
        try:
            # Check the device of model's parameters
            return next(self.parameters()).device
        except StopIteration:
            # If model has no parameters try to check the buffers
            for buffer in self.buffers():
                return buffer.device
        # If there are no buffers, default to CPU
        return torch.device("cpu")
    
    def _reset_hidden_state(self, batch_size=1):
        device = self._get_device()
        self.hx = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        self.cx = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, (self.hx, self.cx) = self.lstm(
            x,
            (
                self.hx,
                self.cx,
            )
        )
        return self.action_head(x), self.value_head(x).squeeze(-1)

def collect_rollouts(
    env,
    policy,
    tasks_per_batch,
    episodes_per_task,
    max_t,
):
    """
    Collect rollouts of a model for `n` tasks with `d` episodes. The result of this function is as follows: \\
        1) A dictionary `trajectories` with the following keys: ["x", "action", "logp", "value", "reward", "done"].
        2) An average task return.
        3) A success matrix `n`x`d` with average values by the first dimension.
    """
    trajectories = []
    task_returns = []
    success_matrix = np.zeros((tasks_per_batch, episodes_per_task))

    device = policy._get_device()
    obs_size = env.observation_space.n
    action_size = env.action_space.n

    for task_idx in range(tasks_per_batch):
        task = env.sample_task()
        env.set_task(task)

        policy._reset_hidden_state()

        prev_action = 0
        prev_reward = 0
        prev_done = 0

        total_task_return = 0

        for episode in range(episodes_per_task):
            obs, _ = env.reset()
            episode_return = 0

            for t in range(max_t):
                x = preprocess_input(
                    obs,
                    prev_action,
                    prev_reward,
                    prev_done,
                    obs_size,
                    action_size,
                    device,
                ).reshape(1, 1, -1) # (, n) -> (1, 1, n)
            
                logits, value = policy(x)
                cat_dist = Categorical(
                    logits=logits.squeeze(1) # (1, 1, d) -> (1, d), remove sequence dimension
                )
                action = cat_dist.sample()
                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                trajectories.append(
                    {
                        "x": x,
                        "action": action,
                        "logp": cat_dist.log_prob(action),
                        "value": value.squeeze(), # (1, 1) -> (,)
                        "reward": reward,
                        "done": done,
                    }
                )

                if reward == 1.0:
                    success_matrix[task_idx, episode] = 1

                episode_return += reward
                prev_reward = reward
                prev_action = action
                prev_done = float(done)
                obs = next_obs

                if done:
                    break
        
            total_task_return += episode_return

        task_returns.append(total_task_return)
    
    return trajectories, np.mean(task_returns), success_matrix.mean(axis=0)

def evaluate(
    env,
    policy,
    episodes_per_task,
    max_t,
    obs_dim,
    action_dim,
    num_tasks=20,
    device="cpu",
):
    """
    Simple evaluation function
    """
    success = np.zeros(episodes_per_task)

    for _ in range(num_tasks):
        task = env.sample_task()
        env.set_task(task)
        policy._reset_hidden_state()

        prev_action = 0
        prev_reward = 0
        prev_done = 0

        for episode in range(episodes_per_task):
            obs, _ = env.reset()

            for _ in range(max_t):
                x = preprocess_input(
                    obs,
                    prev_action,
                    prev_reward,
                    prev_done,
                    obs_dim,
                    action_dim,
                    device,
                ).reshape(1, 1, -1) # (, n) -> (1, 1, n)

                with torch.no_grad():
                    logits, _ = policy(x)
                    action = torch.argmax(logits)
                
                obs, reward, terminated, truncated, _ = env.step(action.item())

                if reward == 1.0:
                    success[episode] += 1
                    break

                if terminated or truncated:
                    break

    return success / num_tasks
