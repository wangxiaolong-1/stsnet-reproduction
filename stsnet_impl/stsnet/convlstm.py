"""ConvLSTM implementation (modernized from official STSNet repo, TUM-LMF original).

Used inside the spatio-temporal transformer block to mine LOCAL spatio-temporal
information from dense time-series Sentinel-2 imagery (paper Eq. 5-9).
"""
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        pad = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=pad, bias=bias)

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(self.conv(combined), self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g          # Eq. 7
        h_next = o * torch.tanh(c_next)  # Eq. 9
        return h_next, c_next

    def init_hidden(self, b, hw, device):
        h, w = hw
        return (torch.zeros(b, self.hidden_dim, h, w, device=device),
                torch.zeros(b, self.hidden_dim, h, w, device=device))


class ConvLSTM(nn.Module):
    """Iterates ConvLSTMCell over the temporal dimension.

    Input : (B, T, C, H, W)  ->  Output : (B, T, C_hidden, H, W)
    When input channel != hidden channel a 1x1 conv adapts the input (paper Sec. 2.2).
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), bias=True):
        super().__init__()
        # inputs are always adapted to hidden_dim first, so the cell always
        # consumes hidden_dim channels
        self.cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size, bias)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.adapt = (nn.Conv2d(input_dim, hidden_dim, 1)
                      if input_dim != hidden_dim else nn.Identity())

    def forward(self, x):
        b, t, _, h, w = x.shape
        state = self.cell.init_hidden(b, (h, w), x.device)
        outs = []
        for k in range(t):
            state = self.cell(self.adapt(x[:, k]), state)
            outs.append(state[0])
        return torch.stack(outs, dim=1)
