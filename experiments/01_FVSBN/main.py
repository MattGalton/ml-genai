from pathlib import Path

from hydra.utils import instantiate
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig
import hydra

from mattg.config.dataset import DatasetLoader
from mattg.training import BinaryAutoRegressiveTask

FILE_DIR = Path(__file__).parent

@hydra.main(
    config_path=str(FILE_DIR.absolute()),
    config_name="config",
    version_base="1.1"
)
def train(cfg: DictConfig):
    task = BinaryAutoRegressiveTask(cfg)
    train_loader, val_loader = DatasetLoader.load(cfg.dataset)
    
    # Create output directory for metrics
    output_dir = Path("outputs") / cfg.logging.name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(output_dir)
    trainer = instantiate(cfg.trainer, logger=logger)

    trainer.fit(task, train_loader, val_loader)
    
    return trainer


if __name__ == "__main__":
    train()


# TODO: Callbacks:
#  * Save model at every checkpoint
#  * Sample and save tiled images at every checkpoint
#  * Add more metrics relevant to gen-ai
