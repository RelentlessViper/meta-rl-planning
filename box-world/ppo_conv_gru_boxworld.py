import os
import random
import time

import draccus
from dataclasses import dataclass
from tqdm import trange

import toymeta
import gymnasium as gym
from gymnasium.envs import register

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from box_world_env import BoxWorld
from conv_gru import ConvGRU

@dataclass
class TrainConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = False
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "rl2-boxworld-meta"
    capture_video: bool = False
    capture_video_every_episode: bool = False
    save_best_model: bool = False

    # Box World arguments
    field_size: int = 8
    goal_length: int = 2
    num_distractor: int = 1
    distractor_length: int = 1
    max_episode_timesteps: int = 1e3
    collect_key: bool = True
    existing_world: np.ndarray = None

    # Algorithm specific arguments
    num_trials: int = 3
    hidden_size: int = 64
    num_layers: int = 1
    total_timesteps: int = 10e6
    learning_rate: float = 1e-3
    num_envs: int = 128
    num_steps: int = 32
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 1
    norm_adv: bool = True
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 10.0
    target_kl: float = None
    
    def __post_init__(self):
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = int(self.total_timesteps // self.batch_size)

        self.env_id = f"Box-World-{self.field_size}x{self.field_size}-{self.goal_length}-{self.num_distractor}-v0"
        self.run_name = f"{self.env_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
        register(
            id=self.env_id,
            entry_point="box_world_env.box_world_env:BoxWorld",
            max_episode_steps=int(self.max_episode_timesteps),
            kwargs=dict(
                n = self.field_size,
                goal_length = self.goal_length,
                num_distractor = self.num_distractor,
                distractor_length = self.distractor_length,
                max_steps = int(self.max_episode_timesteps),
                collect_key = self.collect_key,
                world = self.existing_world,
            ),
        )


def make_env(env_id, idx, capture_video, run_name, capture_video_every_episode):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            if capture_video_every_episode:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: True)
            else:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: obs.astype(np.float32) / 255.0,
            env.observation_space,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    
    return thunk

class ConvGRUAgent(nn.Module):
    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim,
        num_actions,
        num_layers=1,
        batch_first=True,
        bias=True,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        self.conv_gru = ConvGRU(
            input_size=self.input_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            bias=self.bias,
        )
        self.actor = nn.Linear(self.hidden_dim * self.input_size[0] * self.input_size[1], num_actions)
        self.critic = nn.Linear(self.hidden_dim * self.input_size[0] * self.input_size[1], 1)
    
    def get_state(self, x, hidden_state, done):
        # x: [b * t, c, h, w]
        # hidden_state: [l, b, c_hidden, h, w]
        # done: [b * t]
        batch_size = hidden_state.size(1)
        
        # Reshape to [t, b, c, h, w] for sequential processing
        x = x.reshape((-1, batch_size, self.input_dim, *self.input_size))
        done = done.reshape((-1, batch_size))
        
        outputs = []
        for t in range(x.size(0)):
            xt = x[t] # [b, c, h, w]
            dt = done[t].reshape((-1, 1, 1, 1)) # [b, 1, 1, 1] 
            
            for i in range(self.num_layers):
                hidden_state[i] = hidden_state[i] * (1.0 - dt)
            
            # ConvGRU expects [b, t, c, h, w], so unsqueeze T=1
            _, hidden_state = self.conv_gru(xt.unsqueeze(1), hidden_state)
            
            # [b, c_hidden, h, w]
            current_h = hidden_state[-1]
            
            # [b, c_hidden * h * w]
            flattened = current_h.reshape((batch_size, -1))
            outputs.append(flattened)
            
        # Flatten time and batch back together
        hidden_out = torch.cat(outputs, dim=0)
        return hidden_out, hidden_state

    def get_value(self, x, hidden_state, done):
        # x: [b * t, c, h, w]
        # hidden_state: [l, b, c_hidden, h, w]
        # done: [b * t]
        hidden, _ = self.get_state(x, hidden_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, hidden_state, done, action=None):
        # x: [b * t, c, h, w]
        # hidden_state: [l, b, c_hidden, h, w]
        # done: [b * t]
        hidden, hidden_state = self.get_state(x, hidden_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), hidden_state

@draccus.wrap()
def train(args: TrainConfig):
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            config=vars(args),
            name=args.run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, i, args.capture_video, args.run_name, args.capture_video_every_episode) 
        for i in range(args.num_envs)
    ])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = ConvGRUAgent(
        input_size=(args.field_size + 2, args.field_size + 2),
        input_dim=3,
        hidden_dim=args.hidden_size,
        num_actions=envs.single_action_space.n,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.from_numpy(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # next_hidden_state = torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device) # hidden and cell states (see https://youtu.be/8HyCNIVRbSU) Only hidden state since we are using GRU
    #next_hidden_state = agent.get_initial_state(args.batch_size, (args.field_size, args.field_size), device)
    next_hidden_state = agent.conv_gru.init_hidden(args.num_envs)

    # Model saving setup
    if args.save_best_model:
        running_return = 0.0
        best_running_return = -float("inf")
        best_model_path = os.path.join("checkpoints", f"{args.run_name}_best.pt")
        os.makedirs("checkpoints", exist_ok=True)

    for iteration in trange(1, args.num_iterations + 1):
        initial_hidden_state = next_hidden_state.clone()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_hidden_state = agent.get_action_and_value(next_obs, next_hidden_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.from_numpy(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_obs = next_obs

            if "_episode" in infos and args.track:
                for i, has_metrics in enumerate(infos["_episode"]):
                    wandb.log(
                        {
                            f"returns/current_reward": max(infos["episode"]["r"]),
                        },
                        global_step,
                    )
                        
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_hidden_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    initial_hidden_state[:, mbenvinds],
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

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

        if args.save_best_model:
            mean_return = rewards.sum(dim=0).mean().item()
            running_return = 0.9 * running_return + 0.1 * mean_return
            if running_return > best_running_return:
                best_running_return = running_return
                torch.save(
                    {
                        "model_state_dict": agent.state_dict(),
                        "iteration": iteration,
                        "best_running_return": best_running_return,
                    },
                    best_model_path,
                )

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        if args.track:
            wandb.log(
                {
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clip_frac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                },
                global_step,
            )

    envs.close()


if __name__ == "__main__":
    train()