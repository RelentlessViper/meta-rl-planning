import os
import random
import time
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import toymeta

@dataclass
class Args:
    # Experiment specific arguments
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    """The name of the experiment"""
    seed: int = 1
    """The seed of the experiment"""
    torch_deterministic: bool = False
    """If toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """If toggled, cuda will be enabled if possible"""
    track: bool = False
    """If toggled, the experiment will be tracked with WandB"""
    wandb_project_name="rl2-darkroom-meta"
    """The WandB's project name"""
    wandb_entity: str = "king_arthur-org"
    """The entity of WandB's project"""
    capture_video: bool = False
    """If toggled, the video capture would be saved in the `videos` folder"""
    verbose: bool = False
    """If toggled, the episodic returns will be printed in the terminal when the environment is terminated"""

    # Algorithm specific arguments
    env_id: str = "Dark-Room-3x3-v0"
    """ID of the environment in Gymnasium"""
    env_spicific_args: dict[str:any] | None = None
    """Environment specifc arguments"""
    total_timesteps: int = 10_000_000
    """The total amount of timesteps"""
    learning_rate: float  = 2.5e-4
    """The optimizer's learning rate"""
    adam_eps: float = 1e-5
    """The optmizer's eps"""
    num_envs: int = 8
    """The total amount of environments"""
    num_steps: int = 128
    """The number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """If toggled, the learning rate will decrease by annealing for policy and value networks"""
    gamma: float = 0.99
    """The discount factor value"""
    gae_lambda: float = 0.95
    """The strength of GAE"""
    num_minibatches: int = 4
    """The number of mini-batches"""
    update_epochs: int = 4
    """The number of epochs to update the policy"""
    norm_adv: bool = True
    """If toggled, the advantage will be normalized for a stable gradients"""
    clip_coef: float = 0.1
    """The surroggate clipping coefficient"""
    clip_vloss: bool = True
    """If toggled, a clipped loss for the value function will be used"""
    ent_coef: float = 0.01
    """The entropy coefficient value"""
    vf_coef: float = 0.5
    """The coefficient of the value function"""
    max_grad_norm: float = 0.5
    """The maximum norm for the gradient clipping"""
    target_kl: float = None
    """The target KL divergence threshold"""

    # Model specific arguments
    hidden_size: int = 128
    """The hidden_size that will be used in the internal layers"""
    num_layers: int = 1
    """The number of stacked LSTM layers"""

    # Must be filled in the runtime
    batch_size: int = 0
    """The batch size"""
    minibatch_size: int = 0
    """The mini-batch size"""
    num_iter: int = 0
    """The number of iterations"""

def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    run_name,
    env_specific_args = None,
):
    def f():
        if env_specific_args is not None:
            env = gym.make(
                env_id,
                **env_specific_args,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    episode_trigger=lambda x: x % 1000 == 0,
                )
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env
    return f

def one_hot(
    idx,
    size,
):
    x = torch.zeros(size)
    if len(size) == 2:
        for i in range(size[0]):
            x[i][int(idx[i])] = 1.0
    else:
        x[idx] = 1.0
    return x.float()


def one_hot_to_idx(one_hot_tensor):
    if one_hot_tensor.dim() == 2:
        return torch.argmax(one_hot_tensor, dim=1).cpu().numpy()
    else:
        return torch.argmax(one_hot_tensor).item()

def layer_init(
    layer,
    std=np.sqrt(2),
    bias_const=0.0,
):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """
    The main Agent class. \\
    Contains one shared network for Actor & Critic.
    The main structure:
    1) `fc_0` - Linear layer;
    2) `lstm` - N LSTM layers;
    3) `actor` - Linear layer (action-value function);
    4) `critic` - Linear layer (state-value function).
    """
    def __init__(
        self,
        envs,
        hidden_size,
        num_layers=1,
    ):
        super().__init__()
        self.fc_0 = nn.Linear(
            in_features=np.array(envs.single_observation_space.n).prod(),
            out_features=hidden_size,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
        )
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(
            nn.Linear(
                in_features=hidden_size,
                out_features=envs.single_action_space.n,
            ),
            std=0.01,
        )
        self.critic = layer_init(
            nn.Linear(
                in_features=hidden_size,
                out_features=1,
            ),
            std=1,
        )
    
    def get_states(
        self,
        x,
        lstm_state,
        done,
    ):
        hidden_state = torch.relu(self.fc_0(x))
        batch_size = lstm_state[0].shape[1]
        hidden_state = hidden_state.reshape(-1, batch_size, self.lstm.input_size)
        new_hidden_state = []

        for h, d in zip(hidden_state, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).reshape(1, -1, 1) * lstm_state[0], # (1, 8, 1)
                    (1.0 - d).reshape(1, -1, 1) * lstm_state[1], # (1, 8, 1)
                ),
            )
            new_hidden_state += [h]
        new_hidden_state = torch.flatten(
            torch.cat(new_hidden_state), # (1, 8, 128)
            start_dim=0,
            end_dim=1,
        ) # (1 * 8, 128)
        return new_hidden_state, lstm_state
    
    def get_value(
        self,
        x,
        lstm_state,
        done,
    ):
        hidden_state, _ = self.get_states(
            x,
            lstm_state,
            done,
        )
        return self.critic(hidden_state)
    
    def get_action_and_value(
        self,
        x,
        lstm_state,
        done,
        action=None,  
    ):
        hidden_state, lstm_state = self.get_states(
            x,
            lstm_state,
            done,
        )
        logits = self.actor(hidden_state)
        prob_dist = Categorical(logits=logits)
        if action is None:
            action = prob_dist.sample()
        return action, prob_dist.log_prob(action), prob_dist.entropy(), self.critic(hidden_state), lstm_state

# The main execution code
args = Args()
args.env_specific_args = dict(
    random_start=True,
    terminate_on_goal=True,
)
args.track = True
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
args.num_iterations = args.total_timesteps // args.batch_size
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        ) for i in range(args.num_envs)
    ]
)

# Storage
obs = torch.zeros(
    (args.num_steps, args.num_envs) + (int(envs.single_observation_space.n),),
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
).to(device)
values = torch.zeros(
    (args.num_steps, args.num_envs),
).to(device)

# Start
agent = Agent(
    envs,
    args.hidden_size,
    args.num_layers,
).to(device)
optimizer = optim.Adam(
    agent.parameters(),
    args.learning_rate,
    eps=args.adam_eps,
)
global_step = 0
start_time = time.time()
next_obs, _ = envs.reset()
next_obs = one_hot(
    next_obs,
    (args.num_envs, int(envs.single_observation_space.n)),
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

episode_returns = deque(maxlen=100)
episode_lengths = deque(maxlen=100)

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
        action = one_hot_to_idx(action)
        next_obs, reward, terminations, truncations, infos = envs.step(action)
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.Tensor(reward).to(device).reshape(-1)
        next_obs = one_hot(
            next_obs,
            (args.num_envs, envs.single_observation_space.n)
        ).to(device)
        next_done = torch.Tensor(next_done).to(device)

        if args.track:
            if "episode" in infos:
                for r, l in zip(infos["episode"]["r"], infos["episode"]["l"]):
                    episode_returns.append(r)
                    episode_lengths.append(l)
                
                # Log to wandb
                wandb.log(
                    {
                        "charts/mean_episode_return_@_10_steps": np.mean(episode_returns),
                        "charts/mean_episode_length_@10_steps": np.mean(episode_lengths),
                    },
                    global_step,
                )

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
    
    b_obs = obs.reshape((-1,) + (int(envs.single_observation_space.n),))
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
    
envs.close()