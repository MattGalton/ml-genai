import math
import torch
from omegaconf import DictConfig

from mattg.config.optimizer import OptimizerLoader
from mattg.training.tasks.base import BaseTask


class BinaryAutoRegressiveTask(BaseTask):
    def __init__(self, cfg: DictConfig, model: torch.nn.Module):
        super().__init__(cfg)
        self.model = model
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

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

        # flatten for model input (the model expects flattened input)
        x = x.view(x.size(0), -1)

        logits = self.model(x)
        loss = self.loss_fn(logits, x)

        # bpd = nll_base_2 / num_dimensions
        #     = (nll_base_e / math.log(2)) / num_dimensions
        #     = ( (loss * num_dimensions) / math.log(2) ) / num_dimensions
        #     = loss / math.log(2)
        bpd = loss / math.log(2.0)

        perplexity = torch.exp(loss)

        # log per-epoch aggregated scalars (let Lightning handle epoch aggregation)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_bpd", bpd, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_perplexity", perplexity, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": loss}

