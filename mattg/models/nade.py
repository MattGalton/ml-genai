import torch
import torch.nn as nn

from mattg.models.utils.masks import make_nade_masks
from mattg.models.masked_linear import MaskedLinear


class NADE(nn.Module):
    """Neural Autoregressive Density Estimation"""
    def __init__(self, n_visible=28*28, n_hidden=28*28):
        super().__init__()
        m_in, m_out = make_nade_masks(n_visible, n_hidden)
        self.layers = nn.ModuleList([
            MaskedLinear(n_visible, n_hidden, m_in),
            MaskedLinear(n_hidden, n_visible, m_out)
        ])

    def forward(self, x):
        x = torch.relu(self.layers[0](x))
        return self.layers[1](x)
