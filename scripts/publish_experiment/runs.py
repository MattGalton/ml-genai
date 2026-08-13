from pathlib import Path
import re
from typing import Any

from .vars import copy_file, die, read_yaml

def find_runs(experiment_dir: Path) -> list[Path]:
    """
    Finds directories containing metrics.csv.

    Example:

        outputs/MADE_BMNIST/2026-08-09/23-34-46
    """

    outputs = experiment_dir / "outputs"

    if not outputs.exists():
        return []

    return sorted(
        {
            metrics.parent
            for metrics in outputs.rglob("metrics.csv")
        },
        key=lambda p: p.stat().st_mtime,
    )


def find_latest_run(experiment_dir: Path) -> Path:
    runs = find_runs(experiment_dir)

    if not runs:
        die(
            f"No runs containing metrics.csv found under "
            f"{experiment_dir / 'outputs'}"
        )

    return runs[-1]


def find_run(
    experiment_dir: Path,
    run_argument: str | None,
) -> Path:

    if not run_argument:
        return find_latest_run(experiment_dir)

    outputs = experiment_dir / "outputs"

    candidate = outputs / run_argument

    if (candidate / "metrics.csv").exists():
        return candidate

    matches = [
        run
        for run in find_runs(experiment_dir)
        if str(run.relative_to(outputs)).endswith(run_argument)
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        die(
            f"Run '{run_argument}' matched multiple runs:\n"
            + "\n".join(str(m) for m in matches)
        )

    die(f"Could not find run '{run_argument}'")

SAMPLE_PATTERN = re.compile(
    r"samples_epoch_(\d+)\.(png|jpg|jpeg|webp)$",
    re.IGNORECASE,
)


def find_sample_images(run: Path) -> list[dict[str, Any]]:
    plots = run / "plots"

    if not plots.exists():
        return []

    samples: list[dict[str, Any]] = []

    for path in plots.iterdir():
        match = SAMPLE_PATTERN.match(path.name)

        if not match:
            continue

        samples.append(
            {
                "epoch": int(match.group(1)),
                "filename": path.name,
            }
        )

    return sorted(
        samples,
        key=lambda x: x["epoch"],
    )


def copy_plots(
    run: Path,
    destination: Path,
) -> list[str]:

    plots = run / "plots"

    if not plots.exists():
        return []

    copied: list[str] = []

    for path in sorted(plots.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".svg",
        }:
            continue

        copy_file(
            path,
            destination / path.name,
        )

        copied.append(path.name)

    return copied


def find_results_file(run: Path) -> Path | None:
    """
    The reporting callback writes results.yaml into the run directory.
    """

    candidates = [
        run / "results.yaml",
        run / "results.yml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_results(run: Path) -> dict[str, Any]:
    path = find_results_file(run)

    if path is None:
        return {}

    return read_yaml(path)