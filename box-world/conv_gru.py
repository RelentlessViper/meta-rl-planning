import torch
from torch import nn


class ConvGRUCell(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_dim=64,
        kernel_size=(3, 3),
        bias=True,
    ):
        super().__init__()
        self.channels, self.height, self.width = input_shape
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv2d(
            in_channels=self.channels + hidden_dim,
            out_channels=2 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )
        self.conv_candidate = nn.Conv2d(
            in_channels=self.channels + hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, x, hidden_state):
        # x: [b, c, h, w]
        # hidden_state: [b, c_hidden, h, w]
        assert len(x.shape) == len(hidden_state.shape), (
            f"Shapes mismatch. x.shape: {x.shape}, hidden_state.shape: {hidden_state.shape}"
        )
        combined = torch.cat([x, hidden_state], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined_candidate = torch.cat([x, reset_gate * hidden_state], dim=1)
        combined_candidate_conv = self.conv_candidate(combined_candidate)
        candidate_memory = torch.tanh(combined_candidate_conv)

        next_hidden_state = (
            update_gate * hidden_state + (1 - update_gate) * candidate_memory
        )
        return next_hidden_state


class ConvGRU(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_dim=64,
        kernel_size=(3, 3),
        num_layers=1,
        batch_first=False,
        bias=True,
    ):
        super().__init__()
        self.channels, self.height, self.width = input_shape
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cells = []
        for i in range(self.num_layers):
            cell_input_channels = self.channels if i == 0 else self.hidden_dim
            cells.append(
                ConvGRUCell(
                    input_shape=(cell_input_channels, self.height, self.width),
                    hidden_dim=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    bias=self.bias,
                )
            )
        self.cells = nn.ModuleList(cells)

    def init_hidden(self, batch_size):
        return torch.zeros(
            (self.num_layers, batch_size, self.hidden_dim, self.height, self.width),
            device=self.cells[0].conv_gates.weight.device,
        )

    def forward(self, x, hidden_state=None):
        # x: [t, b, c, h, w] or [b, t, c, h, w]
        if self.batch_first:
            # x: [t, b, c, h, w]
            x = x.transpose(0, 1)

        if hidden_state is None:
            hidden_state = self.init_hidden(x.size(1))

        new_hidden_states = []
        inputs = x
        for layer_idx, layer in enumerate(self.cells):
            h = hidden_state[layer_idx]
            outputs = []

            for t in range(x.size(0)):
                h = layer(inputs[t], hidden_state=h)
                outputs.append(h)

            inputs = torch.stack(outputs)
            new_hidden_states.append(h)

        new_hidden_states = torch.stack(new_hidden_states)
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        return inputs, new_hidden_states
