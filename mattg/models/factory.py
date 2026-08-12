from abc import abstractmethod, ABC
from typing import Tuple, List

import lightning as L
import torch.nn as nn

class ModelFactory(ABC):
    @abstractmethod
    def create(self) -> Tuple[nn.Module, List[L.Callback]]:
        ...
    