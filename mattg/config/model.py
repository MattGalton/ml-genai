from dataclasses import dataclass

from hydra.utils import instantiate
from omegaconf import MISSING


@dataclass
class ModelConfig:
    name: str = MISSING
    target: str = MISSING


class ModelLoader:
    @staticmethod
    def load(cfg: ModelConfig):
        return instantiate(cfg.target)
