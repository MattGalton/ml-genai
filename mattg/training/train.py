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
        model_factory = ModelFactoryLoader.load(cfg.model)
        model, model_callbacks = model_factory.create()

        task = BinaryAutoRegressiveTask(cfg, model)

        train_loader, val_loader = DatasetLoader.load(cfg.dataset)

        logger = CSVLogger(save_dir=Path.cwd(), name="", version="")

        callbacks = model_callbacks
        if "callbacks" in cfg.trainer:
            for callback_cfg in cfg.trainer.callbacks:
                callbacks.append(instantiate(callback_cfg))

        trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

        trainer.fit(task, train_loader, val_loader)

        return trainer

    return _train()