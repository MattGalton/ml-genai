import torch
import torch.nn as nn

from mattg.models.factory import ModelFactory
from mattg.models.utils.masks import make_nade_masks
from mattg.models.masked_linear import MaskedLinear


class NADE(nn.Module):
    """Neural Autoregressive Density Estimation"""
    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        m_in, m_out = make_nade_masks(n_visible, n_hidden)
        self.layers = nn.ModuleList([
            MaskedLinear(n_visible, n_hidden, m_in),
            MaskedLinear(n_hidden, n_visible, m_out)
        ])

    def forward(self, x):
        x = torch.relu(self.layers[0](x))
        return self.layers[1](x)


class NADEFactory(ModelFactory):
    def __init__(self, n_visible: int , n_hidden: int):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

    def _create(self):
        return NADE(self.n_visible, self.n_hidden), []

    def _metadata(self):
        n_out = self.n_visible
        architecture = " -> ".join(map(lambda v : str(v), [self.n_visible, self.n_hidden, n_out]))
        return {
            "name": "NADE",
            "description": "Neural Autoregressive Density Estimation",
            "architecture": architecture
        }
