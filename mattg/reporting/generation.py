from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from mattg.sampling import SampleBatch


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


def to_python(value: Any):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(
            value,
            resolve=True,
        )

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


def config_section(cfg: DictConfig | None, name: str):
    if cfg is None or name not in cfg:
        return None

    value = cfg[name]

    return to_python(value)


def without_hydra_keys(value: Any):
    value = to_python(value)

    if isinstance(value, dict):
        return {
            key: without_hydra_keys(child)
            for key, child in value.items()
            if not key.startswith("_")
        }

    if isinstance(value, list):
        return [
            without_hydra_keys(child)
            for child in value
        ]

    return value


def named_target_config(value: Any):
    value = to_python(value)

    if not isinstance(value, dict):
        return value

    target = value.get("target")
    if not isinstance(target, dict):
        return without_hydra_keys(value)

    result = {}
    if "name" in value:
        result["name"] = value["name"]

    result.update(without_hydra_keys(target))
    return result


def dataset_config_summary(value: Any):
    value = to_python(value)

    if not isinstance(value, dict):
        return value

    data_spec = without_hydra_keys(value.get("data_spec", {}))
    if not isinstance(data_spec, dict):
        data_spec = {}

    train = value.get("train", {})
    val = value.get("val", {})

    if "name" not in data_spec and isinstance(train, dict) and "name" in train:
        data_spec["name"] = train["name"]

    split_names = {}
    if isinstance(train, dict) and "name" in train:
        split_names["train"] = train["name"]
    if isinstance(val, dict) and "name" in val:
        split_names["validation"] = val["name"]

    if split_names and len(set(split_names.values())) > 1:
        data_spec["splits"] = split_names

    return data_spec or None


def trainer_config_summary(value: Any):
    value = to_python(value)

    if not isinstance(value, dict):
        return value

    display_keys = (
        "max_epochs",
        "accelerator",
        "devices",
        "precision",
        "log_every_n_steps",
    )

    return {
        key: value[key]
        for key in display_keys
        if key in value
    }


def config_summary(cfg: DictConfig | None):
    section_builders = {
        "dataset": dataset_config_summary,
        "optimizer": named_target_config,
        "trainer": trainer_config_summary,
        "model": named_target_config,
    }

    results = {}
    for section, builder in section_builders.items():
        value = config_section(cfg, section)
        if value is None:
            continue

        value = builder(value)
        if value:
            results[section] = value

    return results


def process_samples(
    samples_path: Path,
    output_plots_dir: Path,
    processed_samples: set[Path] | None = None,
):
    if not samples_path.exists():
        return

    processed_samples = processed_samples if processed_samples is not None else set()
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(samples_path.glob("*.pt")):
        if path in processed_samples:
            continue

        try:
            sample_payload = torch.load(path, weights_only=False)
        except Exception:
            continue

        labels = None
        if isinstance(sample_payload, SampleBatch):
            samples = sample_payload.samples
            labels = sample_payload.labels
        elif isinstance(sample_payload, dict):
            samples = sample_payload.get("samples")
            labels = sample_payload.get("labels")
        else:
            samples = sample_payload

        if samples is None:
            continue

        if labels is not None and not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)

        if isinstance(samples, (list, tuple)):
            samples = torch.stack([torch.as_tensor(sample) for sample in samples])
        elif not isinstance(samples, torch.Tensor):
            samples = torch.as_tensor(samples)

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        num_samples = samples.shape[0]
        if samples.ndim == 2:
            data_dim = samples.shape[1]
            side = int(np.round(np.sqrt(data_dim)))
            if side * side == data_dim:
                images = samples.view(num_samples, side, side)
            elif data_dim == 784:
                images = samples.view(num_samples, 28, 28)
            else:
                images = samples.view(num_samples, 1, data_dim)
        elif samples.ndim == 3:
            images = samples
        else:
            continue

        images_np = images.cpu().numpy()
        vmin = images_np.min()
        vmax = images_np.max()
        if vmax > vmin:
            images_np = (images_np - vmin) / (vmax - vmin)

        cols = int(np.ceil(np.sqrt(num_samples)))
        rows = int(np.ceil(num_samples / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.array(axes).reshape(-1)

        for idx in range(rows * cols):
            ax = axes[idx]
            ax.axis("off")
            if idx < num_samples:
                image = images_np[idx]
                if image.ndim == 2:
                    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(np.transpose(image, (1, 2, 0)))
                if labels is not None and idx < len(labels):
                    ax.set_title(f"Label {int(labels[idx])}", fontsize=10)
            else:
                ax.set_visible(False)

        plt.tight_layout()
        plot_path = output_plots_dir / f"{path.stem}.png"
        plt.savefig(plot_path)
        plt.close()

        metadata = {
            "labels": labels.detach().cpu().tolist() if labels is not None else None,
        }
        with (output_plots_dir / f"{path.stem}.json").open("w", encoding="utf-8") as f:
            import json

            json.dump(metadata, f, indent=2)

        processed_samples.add(path)


def process_metrics(input_metrics_dir: Path, output_plots_dir: Path):
    metrics_path = input_metrics_dir / "metrics.csv"
    if not metrics_path.exists():
        return

    output_plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)
    if "step" not in df.columns or "epoch" not in df.columns:
        return

    steps = df["step"]
    epoch_diff = df["epoch"].diff().fillna(0)
    epoch_boundaries = steps[epoch_diff > 0].values
    metric_cols = [c for c in df.columns if c not in {"step", "epoch", "sample_save"}]

    for col in metric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.dropna().empty:
            continue

        plt.figure(figsize=(8, 4))
        plt.plot(steps, series, marker="o", markersize=3, label=col)

        for boundary in epoch_boundaries:
            plt.axvline(x=boundary, color="gray", linestyle="--", alpha=0.6)

        plt.xlabel("step")
        plt.ylabel(col)
        plt.title(f"{col} over steps")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_plots_dir / f"{col}.png")
        plt.close()


def build_results(
    trainer: L.Trainer,
    pl_module: L.LightningModule,
    cfg: DictConfig | None,
) -> dict[str, Any]:
    training_results = {
        "epochs": trainer.current_epoch,
        "global_step": trainer.global_step,
    }

    final_metrics = {}
    for key, value in trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()

        if isinstance(value, (int, float)):
            final_metrics[key] = value

    best_results = {}
    for callback in trainer.callbacks:
        if not isinstance(callback, L.pytorch.callbacks.ModelCheckpoint):
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

            try:
                best_results["epoch"] = (
                    trainer.checkpoint_callback._parse_ckpt_path(
                        callback.best_model_path
                    ).get("epoch")
                )
            except Exception:
                pass

        break

    config_results = config_summary(cfg)

    results = {
        "training": training_results,
        "metrics": {
            "final": final_metrics,
        },
    }

    if best_results:
        results["metrics"]["best"] = best_results

    if config_results:
        results["config"] = config_results

    return to_python(results)


def write_results(
    output_dir: Path,
    trainer: L.Trainer,
    pl_module: L.LightningModule,
    cfg: DictConfig | None,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "results.yaml"
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            build_results(trainer, pl_module, cfg),
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )

    print(f"Wrote experiment results to {output_path}")
