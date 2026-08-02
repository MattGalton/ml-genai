from typing import List

import torch
import torch.nn as nn

from mattg.models.masked_linear import MaskedLinear


class MADE(nn.Module):
    def __init__(self, layer_sizes: List[int], masks: List[torch.Tensor]):
        super().__init__()

        if len(layer_sizes) - 1 != len(masks):
            raise ValueError(f"Expected {len(layer_sizes)} masks, got {len(masks)}")

        self.layers = nn.ModuleList([
            MaskedLinear(cur_sz, next_sz, mask)
            for cur_sz, next_sz, mask in zip(layer_sizes, layer_sizes[1:], masks)
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def update_masks(self, masks):
        if len(masks) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} masks, got {len(masks)}")
        for layer, mask in zip(self.layers, masks):
            layer.update_mask(mask)
