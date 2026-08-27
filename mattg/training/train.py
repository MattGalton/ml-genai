from pathlib import Path

from hydra.utils import instantiate
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig
import hydra

from mattg.config.dataset import DatasetLoader
from mattg.config.model import ModelFactoryLoader
from mattg.training.tasks import BinaryAutoRegressiveTask

def train_binary_autoregressive(hydra_config_path: Path):

    @hydra.main(
        config_path=str(hydra_config_path.absolute()), config_name="config", version_base="1.1"
    )
    def _train(cfg: DictConfig):
        train_loader, val_loader, data_spec = DatasetLoader.load(cfg.dataset)

        model_factory = ModelFactoryLoader.load(cfg.model)
        model_bundle = model_factory.create(data_spec)

        task = BinaryAutoRegressiveTask(
            cfg=cfg,
            model=model_bundle.model,
            data_spec=data_spec,
            conditioner=model_bundle.conditioner,
        )

        logger = CSVLogger(save_dir=Path.cwd(), name="", version="")

        callbacks = list(model_bundle.callbacks)
        if "callbacks" in cfg.trainer:
            for callback_cfg in cfg.trainer.callbacks:
                callback = instantiate(callback_cfg)
                if hasattr(callback, "cfg") and callback.cfg is None:
                    callback.cfg = cfg
                callbacks.append(callback)

        trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

        trainer.fit(task, train_loader, val_loader)

        return trainer

    return _train()
