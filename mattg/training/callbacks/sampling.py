"""Sampling callbacks for generating and saving model samples during training."""
from pathlib import Path

import torch
from lightning import Callback
from omegaconf import DictConfig

from mattg.conditioner import OneHotConditioner
from mattg.sampling.model_sampler import ModelSampler


class SamplingCallback(Callback):
    def __init__(
        self,
        sampler: ModelSampler,
        sample_every_n_epochs: int = 1,
        output_dir: str = "samples",
        num_samples: int = 16,
        samples_per_label: int | None = None,
        cfg: DictConfig | None = None
    ):
        super().__init__()
        self.sampler = sampler
        self.sample_every_n_epochs = sample_every_n_epochs
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.samples_per_label = samples_per_label
        self.cfg = cfg
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sample(self, pl_module, labels: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = self.num_samples
        data_spec = pl_module.data_spec
        conditioner = pl_module.conditioner
        if labels is None and isinstance(conditioner, OneHotConditioner):
            if data_spec.num_classes is None:
                raise ValueError("Conditional sampling requires data_spec.num_classes")
            if self.samples_per_label is not None:
                labels = torch.arange(data_spec.num_classes).repeat_interleave(self.samples_per_label)
                batch_size = labels.numel()
            else:
                labels = torch.arange(batch_size) % data_spec.num_classes

        return self.sampler.sample(
            model=pl_module.model,
            conditioner=conditioner,
            batch_size=batch_size,
            data_dim=data_spec.data_dim,
            labels=labels,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of every epoch."""
        current_epoch = trainer.current_epoch
        
        if (current_epoch + 1) % self.sample_every_n_epochs != 0:
            return
        
        pl_module.eval()
        with torch.no_grad():
            sample_batch = self.sample(pl_module)
        
        sample_path = (
            self.output_dir / f"samples_epoch_{current_epoch:04d}.pt"
        )
        torch.save(sample_batch.cpu(), sample_path)
        
        if trainer.logger:
            trainer.logger.log_metrics(
                {"sample_save": current_epoch},
                step=current_epoch,
            )
        
        pl_module.train()


class AncestralSamplingCallback(SamplingCallback):
    pass


class OneShotSamplingCallback(SamplingCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sample_every_n_epochs=1, **kwargs)
