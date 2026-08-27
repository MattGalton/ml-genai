from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BinaryBatch:
    x: torch.Tensor
    labels: torch.Tensor | None = None


class BatchAdapter:
    def __call__(self, batch) -> BinaryBatch:
        raise NotImplementedError


class BinaryAutoregressiveBatchAdapter(BatchAdapter):
    def __call__(self, batch) -> BinaryBatch:
        if isinstance(batch, BinaryBatch):
            return batch

        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                x, labels = batch
            elif len(batch) == 1:
                x = batch[0]
                labels = None
            else:
                raise ValueError(f"Expected batch with 1 or 2 elements, got {len(batch)}")
        else:
            x = batch
            labels = None

        return BinaryBatch(
            x=x.view(x.size(0), -1),
            labels=labels,
        )
