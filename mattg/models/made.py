from typing import List, Tuple, Dict, Any

import lightning as L
import torch
import torch.nn as nn

from mattg.models.factory import ModelFactory
from mattg.models.masked_linear import MaskedLinear
from mattg.models.utils.masks import make_made_masks, make_generator


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


class MADEFactory(ModelFactory):
    def __init__(self, layer_sizes: List[int], seed: int = 42, mask_update_frequency: int = 10):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.seed = seed
        self.generator = make_generator(seed)
        self.masks = make_made_masks(self.layer_sizes, self.generator)
        self.mask_update_frequency = mask_update_frequency

    def _create(self) -> Tuple[nn.Module, List[L.Callback]]:
        model = MADE(self.layer_sizes, make_made_masks(self.layer_sizes, self.generator))
        callbacks = [UpdateMaskCallback(self.layer_sizes, self.seed, self.mask_update_frequency)]
        return model, callbacks

    def _metadata(self) -> Dict[Any, Any]:
        architecture = ' -> '.join(map(lambda i : str(i), self.layer_sizes))
        return {
            "name": "MADE",
            "description": "Masked Autoencoder for Distribution Estimation",
            "architecture": architecture,
            "mask_update_frequency": self.mask_update_frequency
        }


class UpdateMaskCallback(L.Callback):
    def __init__(self, layer_sizes, seed, update_frequency):
        super().__init__()
        self.count = 0
        self.update_frequency = update_frequency
        self.layer_sizes = layer_sizes
        self.seed = seed

    def _create_masks(self):
        gen = make_generator(self.seed + self.count)
        return make_made_masks(self.layer_sizes, gen)

    def on_train_epoch_end(self, trainer, pl_module):
        self.count += 1
        if self.count % self.update_frequency != 0:
            return
        pl_module.model.update_masks(self._create_masks())
