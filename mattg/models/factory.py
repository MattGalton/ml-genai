from abc import abstractmethod, ABC
from typing import Tuple, List, Dict, Any

import lightning as L
import torch.nn as nn

class ModelFactory(ABC):

    def create(self) -> Tuple[nn.Module, List[L.Callback]]:
        model, callbacks = self._create()
        setattr(model, "metadata", self._metadata())
        return model, callbacks

    @abstractmethod
    def _create(self) -> Tuple[nn.Module, List[L.Callback]]:
        ...
    
    def _metadata(self) -> Dict[str, Any]:
        """:return: metadata about the model that is written out as consumable results for presentation"""
        return {}
