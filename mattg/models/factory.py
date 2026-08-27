from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Dict, Any

import lightning as L
import torch.nn as nn

from mattg.conditioner import Conditioner
from mattg.datasets.data_spec import DataSpec


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    conditioner: Conditioner
    callbacks: List[L.Callback]


class ModelFactory(ABC):
    def create(self, data_spec: DataSpec | None = None) -> ModelBundle:
        bundle = self._create(data_spec)
        setattr(bundle.model, "metadata", self._metadata())
        return bundle

    @abstractmethod
    def _create(self, data_spec: DataSpec | None = None) -> ModelBundle:
        ...
    
    def _metadata(self) -> Dict[str, Any]:
        """:return: metadata about the model that is written out as consumable results for presentation"""
        return {}
