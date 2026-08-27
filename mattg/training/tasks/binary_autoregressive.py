import math
import torch
from omegaconf import DictConfig

from mattg.config.optimizer import OptimizerLoader
from mattg.conditioner import Conditioner
from mattg.datasets.data_spec import DataSpec
from mattg.training.batch_adapters import BatchAdapter, BinaryAutoregressiveBatchAdapter
from mattg.training.tasks.base import BaseTask


class BinaryAutoRegressiveTask(BaseTask):
    def __init__(self,
                 cfg: DictConfig,
                 model: torch.nn.Module,
                 data_spec: DataSpec,
                 conditioner: Conditioner,
                 batch_adapter: BatchAdapter | None = None):
        super().__init__(cfg)

        self.model = model
        self.data_spec = data_spec
        self.conditioner = conditioner
        self.batch_adapter = batch_adapter or BinaryAutoregressiveBatchAdapter()

        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    @property
    def conditioning(self) -> Conditioner:
        return self.conditioner

    def configure_optimizers(self):
        return OptimizerLoader.load(self.cfg.optimizer, self.model)

    def training_step(self, batch, batch_idx):
        batch = self.batch_adapter(batch)
        
        logits = self.model(self.conditioner.apply(batch.x, batch.labels))
        loss = self.loss_fn(logits, batch.x)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.batch_adapter(batch)

        logits = self.model(self.conditioner.apply(batch.x, batch.labels))
        loss = self.loss_fn(logits, batch.x)

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
