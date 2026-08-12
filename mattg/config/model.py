from dataclasses import dataclass
from typing import Tuple, List

from hydra.utils import instantiate
import lightning as L
from omegaconf import MISSING
import torch.nn as nn


@dataclass
class ModelConfig:
    name: str = MISSING
    target: str = MISSING


class ModelFactoryLoader:
    @staticmethod
    def load(cfg: ModelConfig):
        from mattg.models.factory import ModelFactory

        obj = instantiate(cfg.target)
        if issubclass(type(obj), ModelFactory):
            return obj

        class Factory(ModelFactory):
            def create(self) -> Tuple[nn.Module, List[L.Callback]]:
                return obj, []

        return Factory()

