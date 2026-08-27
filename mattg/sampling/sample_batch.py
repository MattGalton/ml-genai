from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SampleBatch:
    samples: torch.Tensor
    labels: torch.Tensor | None = None

    def cpu(self) -> "SampleBatch":
        return SampleBatch(
            samples=self.samples.cpu(),
            labels=self.labels.cpu() if self.labels is not None else None,
        )
