from dataclasses import dataclass
from pathlib import Path
import yaml

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
        cfg: DictConfig | None = None
    ):
        super().__init__()
        self.cfg = cfg
        self.dirs = OutputDirectoryLayout(Path(dir))
        self._processed_samples = set()

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not self.dirs.root.exists():
            return

        self.process_metrics(self.dirs.root, self.dirs.plots)
        self.process_samples(self.dirs.samples, self.dirs.plots)
        self.write_results(trainer, pl_module)

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

    def write_results(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ):
        """
        Write a concise, presentation-friendly summary of the experiment.

        The resulting results.yaml is intended to be consumed by the
        experiment publisher as well as being useful to a human inspecting
        the run directory.
        """

        def to_python(value):
            """Convert OmegaConf / Torch values into YAML-safe Python values."""
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return value.detach().cpu().item()
                return value.detach().cpu().tolist()

            if isinstance(value, Path):
                return str(value)

            if isinstance(value, dict):
                return {str(k): to_python(v) for k, v in value.items()}

            if isinstance(value, (list, tuple)):
                return [to_python(v) for v in value]

            return value

        def config_section(name: str):
            """
            Extract a useful section from the Hydra config without dumping
            the entire DictConfig / Hydra internals.
            """
            if self.cfg is None or name not in self.cfg:
                return None

            value = self.cfg[name]

            # OmegaConf DictConfig/ListConfig -> regular Python containers.
            try:
                from omegaconf import OmegaConf

                value = OmegaConf.to_container(
                    value,
                    resolve=True,
                )
            except Exception:
                pass

            return to_python(value)

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------

        model = getattr(pl_module, "model", pl_module)
        metadata = getattr(model, "metadata", {})
        model_results = {
            **metadata,
            "class": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------

        training_results = {
            "epochs": trainer.current_epoch + 1,
            "global_step": trainer.global_step,
        }

        # ------------------------------------------------------------------
        # Final metrics
        #
        # callback_metrics contains the epoch-aggregated Lightning metrics
        # after fit has completed.
        # ------------------------------------------------------------------

        final_metrics = {}

        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue

                value = value.detach().cpu().item()

            if isinstance(value, (int, float)):
                final_metrics[key] = value

        # ------------------------------------------------------------------
        # Best validation result
        #
        # ModelCheckpoint exposes the best score and best model path.
        # ------------------------------------------------------------------

        best_results = {}

        for callback in trainer.callbacks:
            if not isinstance(
                callback,
                L.pytorch.callbacks.ModelCheckpoint,
            ):
                continue

            if callback.best_model_score is not None:
                score = callback.best_model_score

                if isinstance(score, torch.Tensor):
                    score = score.detach().cpu().item()

                best_results["metric"] = callback.monitor
                best_results["value"] = score

            if callback.best_model_path:
                best_path = Path(callback.best_model_path)

                best_results["checkpoint"] = best_path.name

                # Lightning checkpoint filenames usually contain the epoch.
                if callback.best_model_path:
                    try:
                        best_results["epoch"] = (
                            trainer.checkpoint_callback._parse_ckpt_path(
                                callback.best_model_path
                            ).get("epoch")
                        )
                    except Exception:
                        pass

            # There is normally only one ModelCheckpoint.
            break

        # ------------------------------------------------------------------
        # Configuration
        #
        # Include the useful experiment-level sections rather than all of
        # Hydra's internal/default metadata.
        # ------------------------------------------------------------------

        config_results = {}

        for section in (
            "dataset",
            "optimizer",
            "trainer",
            "model",
        ):
            value = config_section(section)

            if value is not None:
                config_results[section] = value

        # ------------------------------------------------------------------
        # Assemble
        # ------------------------------------------------------------------

        results = {
            "model": model_results,
            "training": training_results,
            "metrics": {
                "final": final_metrics,
            },
        }

        if best_results:
            results["metrics"]["best"] = best_results

        if config_results:
            results["config"] = config_results


        # ------------------------------------------------------------------
        # Write
        # ------------------------------------------------------------------

        self.dirs.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = self.dirs.root / "results.yaml"

        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                to_python(results),
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )

        print(f"Wrote experiment results to {output_path}")
