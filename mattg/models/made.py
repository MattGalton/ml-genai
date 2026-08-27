from abc import abstractmethod, ABC
from typing import List, Dict, Any

import lightning as L
import torch
import torch.nn as nn

from mattg.conditioner import NoConditioner, OneHotConditioner
from mattg.models.factory import ModelFactory, ModelBundle
from mattg.models.masked_linear import MaskedLinear
from mattg.models.utils.masks import make_made_masks, make_generator, make_made_conditional_masks


class MaskUpdateCallback(L.Callback, ABC):
    def __init__(self, layer_sizes, seed, update_frequency):
        super().__init__()
        self.count = 0
        self.update_frequency = update_frequency
        self.layer_sizes = layer_sizes
        self.seed = seed

    @abstractmethod
    def _create_masks(self):
        ...

    def on_train_epoch_end(self, trainer, pl_module):
        self.count += 1
        if self.count % self.update_frequency != 0:
            return
        pl_module.model.update_masks(self._create_masks())


class MADEMaskUpdateCallback(MaskUpdateCallback):
    def _create_masks(self):
        gen = make_generator(self.seed + self.count)
        return make_made_masks(self.layer_sizes, gen)


class ConditionalMADEMaskUpdateCallback(MaskUpdateCallback):
    def __init__(self, layer_sizes: List[int], num_classes: int, seed: int = 42, update_frequency: int = 10):
        super().__init__(layer_sizes, seed, update_frequency)
        self.num_classes = num_classes

    def _create_masks(self):
        gen = make_generator(self.seed + self.count)
        return make_made_conditional_masks(self.layer_sizes, self.num_classes, gen)


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

    def _create(self, data_spec=None) -> ModelBundle:
        model = MADE(self.layer_sizes, make_made_masks(self.layer_sizes, self.generator))
        callbacks = [
            MADEMaskUpdateCallback(
                self.layer_sizes,
                self.seed,
                self.mask_update_frequency,
            )
        ]
        return ModelBundle(
            model=model,
            conditioner=NoConditioner(),
            callbacks=callbacks,
        )

    def _metadata(self) -> Dict[Any, Any]:
        architecture = ' -> '.join(map(lambda i : str(i), self.layer_sizes))
        return {
            "name": "MADE",
            "description": "Masked Autoencoder for Distribution Estimation",
            "architecture": architecture,
            "mask_update_frequency": self.mask_update_frequency
        }


class ConditionalMADEFactory(ModelFactory):
    def __init__(self, layer_sizes: List[int], num_classes: int | None = None, seed: int = 42, mask_update_frequency: int = 10):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.num_classes = num_classes
        self.resolved_num_classes = num_classes
        self.seed = seed
        self.generator = make_generator(seed)
        self.mask_update_frequency = mask_update_frequency

    def _num_classes(self, data_spec=None) -> int:
        num_classes = self.num_classes if self.num_classes is not None else getattr(data_spec, "num_classes", None)
        if num_classes is None:
            raise ValueError("ConditionalMADEFactory requires num_classes or data_spec.num_classes")
        self.resolved_num_classes = num_classes
        return num_classes

    def model_layer_sizes(self, data_spec=None) -> List[int]:
        num_classes = self._num_classes(data_spec)
        return [self.layer_sizes[0] + num_classes, *self.layer_sizes[1:]]

    def _create(self, data_spec=None) -> ModelBundle:
        num_classes = self._num_classes(data_spec)
        model_layer_sizes = self.model_layer_sizes(data_spec)
        model = MADE(
            model_layer_sizes,
            make_made_conditional_masks(model_layer_sizes, num_classes, self.generator),
        )
        callbacks = [
            ConditionalMADEMaskUpdateCallback(
                model_layer_sizes,
                num_classes,
                self.seed,
                self.mask_update_frequency,
            )
        ]
        return ModelBundle(
            model=model,
            conditioner=OneHotConditioner(num_classes),
            callbacks=callbacks,
        )

    def _metadata(self) -> Dict[Any, Any]:
        architecture = ' -> '.join(map(lambda i : str(i), self.layer_sizes))
        return {
            "name": "MADE",
            "description": "Conditional Masked Autoencoder for Distribution Estimation",
            "architecture": architecture,
            "num_classes": self.resolved_num_classes,
            "mask_update_frequency": self.mask_update_frequency
        }


MADEConditionalFactory = ConditionalMADEFactory
