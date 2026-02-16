import torch
import torch.nn as nn
from torch.distributions import Categorical

import numpy as np

def layer_init(
    layer,
    std=np.sqrt(2),
    bias_const=0.0,
):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class LSTMPPO(nn.Module):
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
        in_features=None,
        num_layers=1,
    ):
        super().__init__()
        if in_features is None:
            in_features = np.array(envs.single_observation_space.n).prod()
        self.fc_0 = nn.Linear(
            in_features=in_features,
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