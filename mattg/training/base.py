"""Base Lightning module template for all tasks."""

from lightning import LightningModule
from omegaconf import DictConfig


class BaseTask(LightningModule):
    """Template for all tasks. Subclass and override as needed."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

    def configure_optimizers(self):
        """Override in subclass."""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """Override in subclass."""
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """Override in subclass. Optional."""
        pass
