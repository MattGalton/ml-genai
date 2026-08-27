from pathlib import Path

import lightning as L
from omegaconf import DictConfig

from mattg.reporting import (
    OutputDirectoryLayout,
    process_metrics,
    process_samples,
    write_results,
)


class ReportingCallback(L.Callback):
    def __init__(
        self,
        dir: str | Path = "outputs",
        cfg: DictConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.dirs = OutputDirectoryLayout(Path(dir))
        self._processed_samples = set()

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not self.dirs.root.exists():
            return

        process_metrics(self.dirs.root, self.dirs.plots)
        process_samples(self.dirs.samples, self.dirs.plots, self._processed_samples)
        write_results(self.dirs.root, trainer, pl_module, self.cfg)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        process_samples(self.dirs.samples, self.dirs.plots, self._processed_samples)
