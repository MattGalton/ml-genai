from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class OutputDirectoryLayout:
    root: Path

    @property
    def hydra(self) -> Path:
        return self.root / ".hydra"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"


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

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not self.dirs.root.exists():
            return

        ReportingCallback.process_metrics(self.dirs.root, self.dirs.plots)
        self.process_samples(self.dirs.samples, self.dirs.plots)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Plot any new samples that have been created during this epoch."""
        self.process_samples(self.dirs.samples, self.dirs.plots)

    def process_samples(self, samples_path: Path, output_plots_dir: Path):
        """Process samples, skipping those that have already been plotted."""
        if not samples_path.exists():
            return

        output_plots_dir.mkdir(parents=True, exist_ok=True)

        for p in sorted(samples_path.glob("*.pt")):
            if p in self._processed_samples:
                continue

            try:
                samples = torch.load(p)
            except Exception:
                continue

            # Convert numpy arrays to torch tensor if needed
            if isinstance(samples, (list, tuple)):
                samples = torch.stack([torch.as_tensor(s) for s in samples])
            elif not isinstance(samples, torch.Tensor):
                samples = torch.as_tensor(samples)

            # Now samples should be a tensor
            if samples.ndim == 1:
                samples = samples.unsqueeze(0)

            # If samples are flat vectors, reshape to square images when possible
            N = samples.shape[0]
            if samples.ndim == 2:
                D = samples.shape[1]
                side = int(np.round(np.sqrt(D)))
                if side * side == D:
                    imgs = samples.view(N, side, side)
                elif D == 784:
                    imgs = samples.view(N, 28, 28)
                else:
                    # Cannot reshape: treat each as 1xD image
                    imgs = samples.view(N, 1, D)
            elif samples.ndim == 3:
                imgs = samples
            else:
                # Unsupported shape
                continue

            # Normalize images to 0-1 for plotting
            imgs_np = imgs.cpu().numpy()
            vmin = imgs_np.min()
            vmax = imgs_np.max()
            if vmax > vmin:
                imgs_np = (imgs_np - vmin) / (vmax - vmin)

            # Create mosaic grid
            cols = int(np.ceil(np.sqrt(N)))
            rows = int(np.ceil(N / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            axes = np.array(axes).reshape(-1)

            for i in range(rows * cols):
                ax = axes[i]
                ax.axis('off')
                if i < N:
                    img = imgs_np[i]
                    if img.ndim == 2:
                        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                    else:
                        # If channels exist, transpose to HWC
                        ax.imshow(np.transpose(img, (1, 2, 0)))
                else:
                    ax.set_visible(False)

            plt.tight_layout()
            out_name = p.stem + ".png"
            out_path = output_plots_dir / out_name
            plt.savefig(out_path)
            plt.close()

            self._processed_samples.add(p)

    @staticmethod
    def process_metrics(input_metrics_dir: Path, output_plots_dir: Path):
        metrics_path = input_metrics_dir / "metrics.csv"
        if not metrics_path.exists():
            return

        output_plots_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(metrics_path)

        # Ensure expected columns exist
        if "step" not in df.columns or "epoch" not in df.columns:
            return

        steps = df["step"]
        epoch_diff = df["epoch"].diff().fillna(0)

        # Determine step positions where epoch increments (epoch boundary markers)
        epoch_boundaries = steps[epoch_diff > 0].values

        # Iterate over all numeric metric columns except control columns
        metric_cols = [c for c in df.columns if c not in {"step", "epoch", "sample_save"}]

        for col in metric_cols:
            # Try to coerce to numeric; skip non-numeric columns
            series = pd.to_numeric(df[col], errors="coerce")
            if series.dropna().empty:
                continue

            plt.figure(figsize=(8, 4))
            plt.plot(steps, series, marker="o", markersize=3, label=col)

            # Draw vertical lines at epoch boundaries
            for boundary in epoch_boundaries:
                plt.axvline(x=boundary, color="gray", linestyle="--", alpha=0.6)

            plt.xlabel("step")
            plt.ylabel(col)
            plt.title(f"{col} over steps")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()

            out_path = output_plots_dir / f"{col}.png"
            plt.savefig(out_path)
            plt.close()
