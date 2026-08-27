"""Base Lightning module template for all tasks."""
from abc import ABC, abstractmethod

from lightning import LightningModule
from omegaconf import DictConfig


class BaseTask(LightningModule, ABC):
    """Template for all tasks. Subclass and override as needed."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

    @abstractmethod
    def configure_optimizers(self):
        ...
    
    @abstractmethod
    def training_step(self, batch, batch_idx):
        ...
    
    def validation_step(self, batch, batch_idx):
        """Optional step for validation"""
        pass