import torch
import torch.nn as nn


class MaskedRNNCell(nn.Module):
    def __init__(self, in_features, out_features, input_mask, bias=True):
        super().__init__()

        self.out_features = out_features

        self.input_linear = nn.Linear(in_features, out_features, bias=bias)
        self.hidden_linear = nn.Linear(out_features, out_features, bias=False)

        self.register_buffer("input_mask", input_mask)

    def forward(self, x, h_prev):
        input_out = nn.functional.linear(
            x, self.input_linear.weight * self.input_mask, self.input_linear.bias
        )
        hidden_out = self.hidden_linear(h_prev)
        return torch.tanh(input_out + hidden_out)

    def h_init(self, batch_size: int, device=None):
        device = device or self.input_mask.device
        return torch.zeros(batch_size, self.out_features, device=device)

    def update_input_mask(self, mask: torch.Tensor):
        self.input_mask.copy_(
            mask.to(device=self.input_mask.device, dtype=self.input_mask.dtype)
        )
