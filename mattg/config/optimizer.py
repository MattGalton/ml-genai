from dataclasses import dataclass

import torch
from hydra.utils import instantiate
from omegaconf import MISSING


@dataclass
class OptimizerConfig:
    name: str = MISSING
    target: str = MISSING


class OptimizerLoader:
    @staticmethod
    def load(cfg: OptimizerConfig, model: torch.nn.Module):
        return instantiate(cfg.target, params=model.parameters())
