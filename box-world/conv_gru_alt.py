import numpy as np

import torch
import torch.nn as nn
from torch.nn import init

class ConvGRUCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(
            in_channels=self.input_size + self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.update_gate = nn.Conv2d(
            in_channels=self.input_size + self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.out_gate = nn.Conv2d(
            in_channels=self.input_size + self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, x, hidden_state=None):
        # x: [b, c, h, w]
        # hidden_state: [b, c_hidden, h, w]
        batch_size = x.size(0)
        spacial_size = x.size()[2:]

        if hidden_state is None:
            state_size = [batch_size, self.hidden_size] + list(spacial_size)
            hidden_state = torch.zeros(state_size).to(self.reset_gate.weight.device)
        
        stacked_inputs = torch.cat([x, hidden_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([x, hidden_state * reset], dim=1)))
        new_state = hidden_state * (1 - update) + out_inputs * update

        return new_state

class ConvGRU(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_dim,
        kernel_size=3,
        num_layers=1,
    ):
        super().__init__()
        self.input_size, self.height, self.width = input_shape
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.hidden_size = hidden_dim
        self.hidden_sizes = [hidden_dim] * num_layers
        self.kernel_sizes = [kernel_size] * num_layers
        
        cells = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]
            
            cell = ConvGRUCell(
                input_size=input_dim,
                hidden_size=self.hidden_sizes[i],
                kernel_size=self.kernel_sizes[i],
            )
            name = "ConvGRUCell_" + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))
        
        self.cells = cells
    
    def init_hidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size, self.height, self.width)).to(self.cells[0].reset_gate.weight.device)
    
    def forward(self, x, hidden_state=None):
        # x: [t, b, c, h, w]
        # hidden_state: [l, b, c_hidden, h, w]
        batch_size = x.size(1)
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
        
        inputs = x
        new_hidden_states = []
        for layer_idx, layer in enumerate(self.cells):
            h = hidden_state[layer_idx]
            outputs = []
            for t in range(x.size(0)):
                h = layer(inputs[t], h)
                outputs.append(h)
            inputs = torch.stack(outputs) # [t, b, c_hidden, h, w]
            new_hidden_states.append(h) # list of [b, c_hidden, h, w] with len = l
        
        return inputs, torch.stack(new_hidden_states)
