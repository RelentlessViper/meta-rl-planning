# %%
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

import wandb
from datetime import datetime

from models import RL2Policy, collect_rollouts, ppo_update, evaluate
from environments import DarkRoomRL2Wrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = dict(
    grid_size=3,
    hidden_dim=128,
    episodes_per_task=5,
    max_steps=20,
    tasks_per_batch=16,
    gamma=0.99,
    lambda_=0.95,
    ppo_eps=0.2,
    entropy_coef=0.02,
    value_coef=0.5,
    lr=3e-4,
    updates=1000,
)

wandb.init(
    project="rl2-darkroom-meta",
    config=config,
)

GRID_SIZE = 3
OBS_DIM = GRID_SIZE * GRID_SIZE
ACTION_DIM = 5

NUM_EPOCHS = 1000
EPISODES_PER_TASK = 5
MAX_T = 20
TASKS_PER_BATCH = 16

GAMMA = 0.99
LAMBDA = 0.95
PPO_EPS = 0.2
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5
LR = 3e-4

# %%
env = DarkRoomRL2Wrapper(gym.make("Dark-Room-3x3-v0"))
policy = RL2Policy(OBS_DIM, ACTION_DIM).to(device)
optimizer = optim.Adam(policy.parameters(), lr=LR)

#for update in range(NUM_EPOCHS):
#    trajectories = collect_rollouts(
#        env,
#        policy,
#        TASKS_PER_BATCH,
#       EPISODES_PER_TASK,
#       MAX_T,
#        OBS_DIM,
#        ACTION_DIM,
#    )
#    ppo_update(
#        policy,
#        optimizer,
#        trajectories,
#        PPO_EPS,
#        VALUE_COEF,
#        ENTROPY_COEF,
#        GAMMA,
#        LAMBDA,
#    )
#
#    if update % 20 == 0:
#        print(f"Update {update}")

# %%
best_success = 0

for update in range(config["updates"]):

    trajectories, mean_return, success_per_episode = collect_rollouts(
        env,
        policy,
        TASKS_PER_BATCH,
        EPISODES_PER_TASK,
        MAX_T,
        OBS_DIM,
        ACTION_DIM,
    )

    metrics = ppo_update(
        policy,
        optimizer,
        trajectories,
        PPO_EPS,
        VALUE_COEF,
        ENTROPY_COEF,
        GAMMA,
        LAMBDA,
    )

    # Evaluation (lightweight)
    eval_success = evaluate(
        env,
        policy,
        EPISODES_PER_TASK,
        MAX_T,
        OBS_DIM,
        ACTION_DIM,
        device,
    )

    # Logging
    wandb.log(
        {
            "update": update,
            "mean_task_return": mean_return,
            "eval_success_ep1": eval_success[0],
            "eval_success_ep2": eval_success[1],
            "eval_success_ep3": eval_success[2],
            "eval_success_ep4": eval_success[3],
            "eval_success_ep5": eval_success[4],
            "train_success_ep1": success_per_episode[0],
            "train_success_ep2": success_per_episode[1],
            "train_success_ep3": success_per_episode[2],
            "train_success_ep4": success_per_episode[3],
            "train_success_ep5": success_per_episode[4],
            **metrics
        }
    )

    # Save latest every 50 updates
    #if update % 50 == 0:
    #    save_checkpoint(policy, optimizer, update, tag="latest")

    # Save best model
    #avg_eval_success = eval_success.mean()
    #if avg_eval_success > best_success:
    #    best_success = avg_eval_success
    #    save_checkpoint(policy, optimizer, update, tag="best")
