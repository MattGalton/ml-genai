import torch
from hydra.utils import instantiate
from omegaconf import MISSING

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataLoaderConfig:
    _target_: str = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING


@dataclass
class DatasetImplConfig:
    _target_: str = MISSING


@dataclass
class SingleDatasetConfig:
    name: str = MISSING
    dataset: DatasetImplConfig = field(default_factory=DatasetImplConfig)
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)


@dataclass
class DatasetConfig:
    train: SingleDatasetConfig = None
    val: Optional[SingleDatasetConfig] = None


class DatasetLoader:
    @staticmethod
    def load(cfg: DatasetConfig) -> tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
        train_dataset = DatasetLoader._load_single(cfg.train, train=True)
        val_dataset = DatasetLoader._load_single(cfg.val, train=False) if cfg.val else None
        return train_dataset, val_dataset

    @staticmethod
    def _load_single(cfg: SingleDatasetConfig, train: bool) -> torch.utils.data.DataLoader:
        dataset = instantiate(cfg.dataset, train=train)
        loader = instantiate(cfg.loader, dataset=dataset)
        return loader
