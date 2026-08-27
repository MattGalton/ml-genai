import torch

from mattg.conditioner import Conditioner
from mattg.sampling.model_sampler import ModelSampler
from mattg.sampling.sample_batch import SampleBatch


class AncestralSampler(ModelSampler):
    @torch.no_grad()
    def sample(
        self,
        model,
        conditioner: Conditioner,
        batch_size: int,
        data_dim: int,
        labels: torch.Tensor | None = None,
    ) -> SampleBatch:
        device = next(model.parameters()).device
        x = torch.zeros(batch_size, data_dim, device=device)
        labels = labels.to(device) if labels is not None else None

        for dim_idx in range(data_dim):
            model_input = conditioner.apply(x, labels)
            logits = model(model_input)
            probs = torch.sigmoid(logits[:, dim_idx])
            x[:, dim_idx] = torch.bernoulli(probs)
        return SampleBatch(samples=x, labels=labels)
