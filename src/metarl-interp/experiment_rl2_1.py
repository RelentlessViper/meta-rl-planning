# %%
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import toymeta
from models import Args, LSTMPPO, one_hot, one_hot_to_idx
from environments import RenderScalerDarkRoom, RL2DarkRoom

def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    run_name,
    env_specific_args=None,
    trials_per_episode=None,
):
    def f():
        if env_specific_args is not None:
            env = gym.make(
                env_id,
                **env_specific_args,
            )
        else:
            env = gym.make(env_id)
        env = RL2DarkRoom(
            env,
            trials_per_episode=trials_per_episode,
        )
        env = RenderScalerDarkRoom(
            env,
            scale_factor=100,
            draw_lines=True,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"/home/jovyan/gubaidullin_0/videos/{run_name}",
                    episode_trigger=lambda x: x % 100 == 0,
                )
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env
    return f

def make_model_input(
    obs,
    actions,
    rewards,
    trial_dones,
):
    return torch.cat(
        [
            obs,
            actions,
            rewards,
            trial_dones,
        ]
    )

# Set up the training args
args = Args(
    exp_name="8-envs_2-trials_1e+6-steps",
    num_envs=8,
    total_timesteps=1_000_000,
    num_minibatches=4,
    env_specific_args=dict(
        random_start=False,
        terminate_on_goal=False,
    ),
    track=True,
    capture_video=True,
)
args.trials_per_episode = 2
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.num_envs // args.num_minibatches)
args.num_iterations = int(args.total_timesteps // args.batch_size)
run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"

if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

envs = gym.vector.SyncVectorEnv(
    [
        make_env(
            args.env_id,
            args.seed,
            i,
            args.capture_video,
            run_name,
            args.env_specific_args,
            args.trials_per_episode,
        ) for i in range(args.num_envs)
    ]
)

# Storage
obs = torch.zeros(
    (args.num_steps, args.num_envs) + (int(envs.single_observation_space.n + envs.single_action_space.n + 2),), # Since we concat observations with previous_actions, rewards and trial dones (not episodic dones). So the hidden dim = observation_size + action_size + 2
).to(device)
actions = torch.zeros(
    (args.num_steps, args.num_envs) + (int(envs.single_action_space.n),),
    dtype=torch.long,
).to(device)
logprobs = torch.zeros(
    (args.num_steps, args.num_envs),
).to(device)
rewards = torch.zeros(
    (args.num_steps, args.num_envs),
).to(device)
dones = torch.zeros(
    (args.num_steps, args.num_envs),
).to(device) # These are the episodic dones
values = torch.zeros(
    (args.num_steps, args.num_envs),
).to(device)

# Agent and optimizer
in_features = envs.single_observation_space.n + envs.single_action_space.n + 2
agent = LSTMPPO(
    envs,
    hidden_size=args.hidden_size,
    in_features=in_features,
    num_layers=args.num_layers,
).to(device)
optimizer = optim.Adam(
    agent.parameters(),
    args.learning_rate,
    eps=args.adam_eps,
)

# Initialize our observations and LSTM states
global_step = 0
start_time = time.time()
next_obs, _ = envs.reset()
next_obs = one_hot(
    next_obs,
    (args.num_envs, int(envs.single_observation_space.n)),
).to(device)
next_actions = torch.zeros((args.num_envs, envs.single_action_space.n)).to(device)
next_rewards = torch.zeros((args.num_envs, 1)).to(device)
next_trial_dones = torch.zeros((args.num_envs, 1)).to(device)
next_obs = torch.cat(
    [
        next_obs,
        next_actions,
        next_rewards,
        next_trial_dones,
    ],
    dim=1,
).to(device)
next_done = torch.zeros(args.num_envs).to(device)
next_lstm_state = (
    torch.zeros(
        agent.lstm.num_layers,
        args.num_envs,
        agent.lstm.hidden_size,
    ).to(device),
    torch.zeros(
        agent.lstm.num_layers,
        args.num_envs,
        agent.lstm.hidden_size,
    ).to(device)
)
# The initialization of the whole training loop
for iteration in range(1, args.num_iterations + 1):
    # Initialize the hidden state and anneal the learning rate
    init_lstm_state = (
        next_lstm_state[0].clone(),
        next_lstm_state[1].clone(),
    )
    if args.anneal_lr:
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        lr_now = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lr_now
    
    # Rollout
    for step in range(args.num_steps):
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                next_obs,
                next_lstm_state,
                next_done,
            )
            action = one_hot(
                action.cpu().numpy(),
                (args.num_envs, int(envs.single_action_space.n))
            ).to(device)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # Execute an action
        action_encoded = action.clone()
        action = one_hot_to_idx(action)
        next_obs, reward, terminations, truncations, infos = envs.step(action)

        # Reshape rewards, 
        next_done = np.logical_or(terminations, truncations)

        # Log rewards for each goal position
        next_trial_dones = infos["trial_done"]
        if args.track and "cumulative_reward_per_trial" in infos:
            for idx, trial_done in enumerate(next_trial_dones):
                if trial_done:
                    goal_pos = envs.envs[idx].unwrapped.goal_pos
                    trial_num = infos["current_trial"][idx]
                    wandb.log(
                        {
                            f"trial_{trial_num}/return_{goal_pos[0]}_{goal_pos[1]}": infos["cumulative_reward_per_trial"][idx][trial_num],
                        },
                        global_step,
                    )

        rewards[step] = torch.Tensor(reward).to(device).reshape(-1)
        next_obs = one_hot(
            next_obs,
            (args.num_envs, envs.single_observation_space.n)
        ).to(device)
        reward = torch.Tensor(reward).to(device).reshape((-1, 1))
        next_trial_dones = torch.Tensor(infos["trial_done"]).to(device).reshape((-1, 1))
        next_obs = torch.cat(
            [
                next_obs,
                action_encoded,
                reward,
                next_trial_dones,
            ],
            dim=1,
        ).to(device)
        next_done = torch.Tensor(next_done).to(device)

    # Bootstrapping + GAE
    with torch.no_grad():
        next_value = agent.get_value(
            next_obs,
            next_lstm_state,
            next_done,
        ).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        last_gae_lambda = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            td_error = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lambda = td_error + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lambda
        returns = advantages + values
    
    b_obs = obs.reshape((-1,) + (int(envs.single_observation_space.n + envs.single_action_space.n + 2),))
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + (int(envs.single_action_space.n),))
    #b_actions = actions.reshape(-1)
    b_dones = dones.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the agent
    envs_per_batch = args.num_envs // args.num_minibatches
    env_inds = np.arange(args.num_envs)
    flat_inds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
    clip_fracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(env_inds)

        for start in range(0, args.num_envs, envs_per_batch):
            end = start + envs_per_batch
            mb_env_inds = env_inds[start:end]
            mb_inds = flat_inds.ravel()

            _, new_logprob, entropy, new_value, _ = agent.get_action_and_value(
                b_obs[mb_inds],
                (
                    init_lstm_state[0][:, mb_env_inds],
                    init_lstm_state[1][:, mb_env_inds]
                ),
                b_dones[mb_inds],
                torch.Tensor(one_hot_to_idx(b_actions.long())).long().to(device)[mb_inds],
            )
            logratio = new_logprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clip_fracs += [
                    (
                        (ratio - 1.0).abs() > args.clip_coef
                    ).float().mean().item()
                ]
            
            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            
            # Policy loss
            pg_loss_1 = -mb_advantages * ratio
            pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

            # Value loss
            new_value = new_value.reshape(-1)
            if args.clip_vloss:
                v_loss_unclipped = (new_value - b_returns) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(new_value - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()
            
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
        
        if args.target_kl is not None and approx_kl > args.target_kl:
            break
    
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    if args.track:
        wandb.log(
            {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clip_frac": np.mean(clip_fracs),
                "losses/explained_variance": explained_var,
                "charts/SPS": int(global_step / (time.time() - start_time)),
            },
            global_step,
        )