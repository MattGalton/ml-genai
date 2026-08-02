import torch
import torch.nn as nn


class RNN(nn.Module):
    """Simple recurrent model with a 1 -> hidden -> 1 step."""

    def __init__(self, n_hidden=28 * 28):
        super().__init__()
        self.n_hidden = n_hidden
        self.input_linear = nn.Linear(1, n_hidden, bias=True)
        self.hidden_linear = nn.Linear(n_hidden, n_hidden, bias=False)
        self.out = nn.Linear(n_hidden, 1, bias=True)

    def forward(self, x, h_prev):
        h_next = torch.tanh(self.input_linear(x) + self.hidden_linear(h_prev))
        return self.out(h_next), h_next

    def h_init(self, batch_size: int, device=None):
        device = device or self.input_linear.weight.device
        return torch.zeros(batch_size, self.n_hidden, device=device)
