from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Conditioner(ABC):
    @abstractmethod
    def apply(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        ...


class NoConditioner(Conditioner):
    def apply(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return x


class OneHotConditioner(Conditioner):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def apply(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if labels is None:
            raise ValueError("OneHotConditioner requires labels")

        y = F.one_hot(
            labels,
            num_classes=self.num_classes,
        ).to(device=x.device, dtype=x.dtype)

        return torch.cat([x, y], dim=-1)
