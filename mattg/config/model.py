from dataclasses import dataclass

from hydra.utils import instantiate
from omegaconf import MISSING


@dataclass
class ModelConfig:
    name: str = MISSING
    target: str = MISSING


class ModelFactoryLoader:
    @staticmethod
    def load(cfg: ModelConfig):
        from mattg.models.factory import ModelFactory

        obj = instantiate(cfg.target)
        if issubclass(type(obj), ModelFactory):
            return obj

        class Factory(ModelFactory):
            def _create(self, data_spec=None):
                from mattg.conditioner import NoConditioner
                from mattg.models.factory import ModelBundle

                return ModelBundle(
                    model=obj,
                    conditioner=NoConditioner(),
                    callbacks=[],
                )

        return Factory()
