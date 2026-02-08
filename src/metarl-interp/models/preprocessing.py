import torch

def one_hot(idx, size):
    x = torch.zeros(size)
    x[idx] = 1.0
    return x.float()

def preprocess_input(
    obs,
    prev_action,
    prev_reward,
    prev_done,
    obs_size,
    action_size,
    device="cpu",
):
    return torch.cat(
        [
            one_hot(obs, obs_size),
            one_hot(prev_action, action_size),
            torch.tensor(
                [
                    prev_reward,
                    prev_done,
                ]
            ).float()
        ]
    ).to(device)

def compute_advantages(
    trajectories,
    gamma,
    lambda_t,
):
    """
    Generalized Advantage Estimation to balance tradeoff between bias and variance
    """
    advantages = []
    returns = []

    gae = 0
    next_value = 0

    for step in reversed(trajectories):
        td_error = step["reward"] + gamma * next_value * (1 - step["done"]) - step["value"]
        gae = td_error + gamma * lambda_t * (1 - step["done"]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + step["value"])
        next_value = step["value"]
    
    return torch.stack(advantages), torch.stack(returns)
