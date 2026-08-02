import torch
from omegaconf import DictConfig

from mattg.config.model import ModelLoader
from mattg.config.optimizer import OptimizerLoader
from mattg.training.base import BaseTask


class BinaryAutoRegressiveTask(BaseTask):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = ModelLoader.load(cfg.model)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        """Create optimizer from cfg."""
        return OptimizerLoader.load(self.cfg.optimizer, self.model)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        
        logits = self.model(x)
        loss = self.loss_fn(logits, x)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        
        logits = self.model(x)
        loss = self.loss_fn(logits, x)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss
