from abc import abstractmethod, ABC

import torch

from mattg.conditioner import Conditioner
from mattg.sampling.sample_batch import SampleBatch


class ModelSampler(ABC):
    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        model,
        conditioner: Conditioner,
        batch_size: int,
        data_dim: int,
        labels: torch.Tensor | None = None,
    ) -> SampleBatch:
        ...
