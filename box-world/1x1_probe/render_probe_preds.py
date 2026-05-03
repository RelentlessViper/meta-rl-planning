import os
import time
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from conv_gru import ConvGRU

import draccus
from dataclasses import dataclass
from tqdm import trange
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.envs import register
from box_world_env import BoxWorld, RevealChestContentsWrapper, RL2BoxWorld, ProbeRenderWrapper

VIDEO_DIR = "./videos_boxworld"
os.makedirs(VIDEO_DIR, exist_ok=True)

@dataclass
class ProbeRecordingConfig:
    dataset_path: str = None
    seed: int = 1
    model_checkpoint_path: str = None
    probe_checkpoint_path: str = None
    save_video_path: str = None
    fps: int = 8
    hidden_size: int = 32
    num_layers: int = 1
    num_trials: int = 3
    num_episodes: int = 20
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
        self.run_name = (
            f"{self.env_id}__{int(time.time())}__{self.seed}"
        )
        
        self.save_video_path = f"videos/{self.run_name}" if self.save_video_path is None else self.save_video_path + f"/{self.run_name}"
        
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

def make_env(env_id, num_trials, probe, fps, save_video_path):
    env = gym.make(env_id, collect_key=False, render_mode="rgb_array")
    env = RevealChestContentsWrapper(env, render_original_env=False)
    env = RL2BoxWorld(env, trials_per_episode=num_trials)
    env = ProbeRenderWrapper(env, probe, fps=fps)
    env = RecordVideo(
        env,
        video_folder=save_video_path,
        episode_trigger=lambda x: True,
        name_prefix=f"box-world-probe",
        fps=fps,
    )
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

@draccus.wrap()
def render_probe(args: ProbeRecordingConfig):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    with open(args.probe_checkpoint_path, "rb") as f:
        probe = pickle.load(f)

    env = make_env(args.env_id, args.num_trials, probe, args.fps, args.save_video_path)

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

    obs, info = env.reset(seed=0)

    done = False
    truncated = False

    # Example agent loop
    with torch.no_grad():
        for episode in trange(args.num_episodes, desc="Inference"):
            next_obs, info = env.reset()
            next_obs = torch.from_numpy(next_obs).to(device)
            next_done = torch.zeros(1).to(device)
            next_hidden_state = agent.rnn.init_hidden(1) # Since we are using only one env

            for step in range(env.spec.max_episode_steps * args.num_trials):
                action, _, _, _, next_hidden_state = agent.get_action_and_value(
                    next_obs,
                    next_hidden_state,
                    next_done,
                    argmax=True,
                )
                probe_hidden_state = next_hidden_state[-1].squeeze(0).clone()
                env.env.set_hidden_state(probe_hidden_state)

                next_obs, reward, terminated, truncated, info = env.step(action.item())
                next_done = int(terminated or truncated)
                next_obs, next_done = torch.from_numpy(next_obs).to(device), torch.Tensor([next_done]).to(device)

                if terminated or truncated:
                    break

if __name__ == "__main__":
    render_probe()