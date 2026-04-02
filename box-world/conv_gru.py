import torch
from torch import nn

class ConvGRUCell(nn.Module):
    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim = 64,
        kernel_size = (3, 3),
        bias = True,
    ):
        super().__init__()
        self.height, self.width = input_size
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * self.hidden_dim, # Since we use update and reset gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_candidate = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )
    
    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_dim, self.height, self.width), device=self.conv_gates.weight.device)

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

        next_hidden_state = (1 - update_gate) * candidate_memory + update_gate * hidden_state
        
        return next_hidden_state

class ConvGRU(nn.Module):
    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim = 64,
        kernel_size = (3, 3),
        num_layers = 1,
        batch_first = True,
        bias = True,
    ):
        super().__init__()

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cells = []
        for i in range(self.num_layers):
            if i == 0:
                cell_input_dim = input_dim
            else:
                cell_input_dim = hidden_dim[i - 1]
            cells.append(
                ConvGRUCell(
                    input_size=input_size,
                    input_dim=cell_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )
        self.cells = nn.ModuleList(cells)
    
    def forward(self, x, hidden_state=None, return_all_layers=False):
        # x: [b, t, c, h, w] or [t, b, c, h, w] if batch_first == True
        if not self.batch_first:
            # [t, b, c, h, w] -> [b, t, c, h, w]
            x = x.permute(1, 0, 2, 3, 4)

        if hidden_state is None:
            hidden_state = self.init_hidden(x.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = x.size(1)
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # x: [b,t,c,h,w] h: [b, c_hidden, h, w] 
                h = self.cells[layer_idx](x=cur_layer_input[:, t, :, :, :], hidden_state=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return torch.stack(layer_output_list), torch.stack(last_state_list)
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cells[i].init_hidden(batch_size))
        return torch.stack(init_states)

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param