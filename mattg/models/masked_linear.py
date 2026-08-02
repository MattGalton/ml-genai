import torch
import torch.nn as nn


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)

    def update_mask(self, mask: torch.Tensor):
        self.mask.copy_(mask.to(device=self.mask.device, dtype=self.mask.dtype))
