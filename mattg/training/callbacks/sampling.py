"""Sampling callbacks for generating and saving model samples during training."""

from pathlib import Path

import torch
from lightning import Callback
from omegaconf import DictConfig


class BaseSamplingCallback(Callback):
    def __init__(
        self,
        sample_every_n_epochs: int = 1,
        output_dir: str = "samples",
        num_samples: int = 16,
        cfg: DictConfig | None = None
    ):
        super().__init__()
        self.sample_every_n_epochs = sample_every_n_epochs
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.cfg = cfg
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sample(self, pl_module, input_dim: int) -> torch.Tensor:
        """Override in subclass. Returns samples tensor."""
        raise NotImplementedError

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of every epoch."""
        current_epoch = trainer.current_epoch
        
        if (current_epoch + 1) % self.sample_every_n_epochs != 0:
            return
        
        pl_module.eval()
        with torch.no_grad():
            input_dim = getattr(
                pl_module.model,
                "input_dim",
                784,
            )
            samples = self.sample(pl_module, input_dim)
        
        sample_path = (
            self.output_dir / f"samples_epoch_{current_epoch:04d}.pt"
        )
        torch.save(samples.cpu(), sample_path)
        
        if trainer.logger:
            trainer.logger.log_metrics(
                {"sample_save": current_epoch},
                step=current_epoch,
            )
        
        pl_module.train()


class AncestralSamplingCallback(BaseSamplingCallback):
    """Ancestral sampling: dimension-by-dimension conditioning."""

    def sample(self, pl_module, input_dim: int) -> torch.Tensor:
        batch_size = self.num_samples
        samples = torch.zeros(batch_size, input_dim, device=pl_module.device)
        
        for dim_idx in range(input_dim):
            logits = pl_module.model(samples)
            probs = torch.sigmoid(logits[:, dim_idx])
            samples[:, dim_idx] = torch.bernoulli(probs)
        
        return samples


class OneShotSamplingCallback(BaseSamplingCallback):
    """One-shot sampling: all dimensions in parallel."""

    def sample(self, pl_module, input_dim: int) -> torch.Tensor:
        batch_size = self.num_samples
        samples = torch.zeros(batch_size, input_dim, device=pl_module.device)
        
        logits = pl_module.model(samples)
        probs = torch.sigmoid(logits)
        samples = torch.bernoulli(probs)
        
        return samples
