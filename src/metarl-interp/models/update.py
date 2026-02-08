import torch
import torch.nn as nn
from torch.distributions import Categorical

from .preprocessing import compute_advantages

def ppo_update(
    policy,
    optimizer,
    trajectories,
    ppo_eps,
    value_coef,
    entropy_coef,
    gamma,
    lambda_t,
):
    """
    Perform a single PPO update on the set of trajectories for a given policy
    """
    advantages, returns = compute_advantages(
        trajectories,
        gamma,
        lambda_t,
    )

    # Normalize the advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e+8)

    policy_loss = 0
    value_loss = 0
    entropy = 0
    num_trajectories = len(trajectories)

    for step, advantage, return_t in zip(trajectories, advantages, returns):
        logits, value = policy(step["x"])
        cat_dist = Categorical(
            logits=logits.squeeze(0), # (1, 1, d) -> (1, d), remove sequence dimension
        )
        new_logp = cat_dist.log_prob(step["action"])
        ratio = torch.exp(new_logp - step["logp"])
        clipped = torch.clamp(ratio, 1 - ppo_eps, 1 + ppo_eps)

        policy_loss += -torch.min(ratio * advantage, clipped * advantage)
        value_loss += (return_t - value.squeeze()) ** 2 # (1, 1) -> (,). Square it since we calculate MSE
        entropy += cat_dist.entropy()
    
    policy_loss /= num_trajectories
    value_loss /= num_trajectories
    entropy /= num_trajectories

    loss = (policy_loss / num_trajectories + value_coef * value_loss - entropy_coef * entropy.mean())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss_total": loss.item(),
        "loss_policy": policy_loss.item(),
        "loss_value": value_loss.item(),
        "entropy": entropy.item(),
        "return_mean": returns.mean().item()
    }