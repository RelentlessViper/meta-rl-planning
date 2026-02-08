import gymnasium as gym
import torch
import wandb
import os

from environments import DarkRoomRL2Wrapper
from models import RL2LSTMPolicy, collect_rollouts, ppo_update, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    wandb.init(project="rl2-darkroom-meta")

    config = wandb.config

    run_name = wandb.run.name
    save_dir = os.path.join("/home/jovyan/gubaidullin_0", run_name)
    os.makedirs(save_dir, exist_ok=True)

    env = DarkRoomRL2Wrapper(
        gym.make(f"Dark-Room-3x3-v0")
    )

    policy = RL2LSTMPolicy(
        env.observation_space.n,
        env.action_space.n,
        hidden_size=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(config.device)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.lr
    )

    best_eval = -float("inf")

    for update in range(config.updates):

        trajectories, mean_return, success_per_episode = collect_rollouts(
            env,
            policy,
            config.tasks_per_batch,
            config.episodes_per_task,
            config.max_t,
        )

        metrics = ppo_update(
            policy,
            optimizer,
            trajectories,
            config.ppo_eps,
            config.value_coef,
            config.entropy_coef,
            config.gamma,
            config.lambda_t,
        )

        wandb.log({
            "update": update,
            "train/mean_return": mean_return,
            "train/success_ep1": success_per_episode[0],
            "train/success_last_ep": success_per_episode[-1],
            **{f"train/{k}": v for k, v in metrics.items()}
        })

        if update % config.eval_interval == 0:

            eval_success = evaluate(
                env,
                policy,
                config.episodes_per_task,
                config.max_t,
                env.observation_space.n,
                env.action_space.n,
                device=config.device,
            )

            final_success = eval_success[-1]

            wandb.log({
                "eval/success_ep1": eval_success[0],
                "eval/success_last_ep": final_success,
            })

            if final_success > best_eval:
                best_eval = final_success

                torch.save(
                    {
                        "model_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "update": update,
                    },
                    os.path.join(save_dir, "best_model.pt"),
                )

    wandb.finish()

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "eval/success_last_ep",
        "goal": "maximize"
    },
    "parameters": {
        "hidden_dim": {
            "values": [64, 128, 256]
        },
        "num_layers": {
            "values": [1, 2, 3]
        },
        "lr": {
            "values": [1e-4, 3e-4, 1e-3]
        },
        "tasks_per_batch": {
            "values": [8, 16, 32]
        },
        "episodes_per_task": {
            "values": [3, 5, 8]
        },

        # Fixed parameters
        "grid_size": {"value": 3},
        "max_t": {"value": 20},
        "gamma": {"value": 0.99},
        "lambda_t": {"value": 0.95},
        "ppo_eps": {"value": 0.2},
        "entropy_coef": {"value": 0.02},
        "value_coef": {"value": 0.5},
        "updates": {"value": 1000},
        "eval_interval": {"value": 20},
        "checkpoint_interval": {"value": 100},
        "device": {"value": "cuda"},
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="rl2-darkroom-meta")
    wandb.agent(sweep_id, function=train)
