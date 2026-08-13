import json
from pathlib import Path
import re
from typing import Any

from .metrics import read_metrics, find_best_checkpoint
from .rendering import generate_space_readme, generate_index_html
from .runs import copy_plots, find_sample_images, find_results_file, load_results
from .vars import ROOT, EXPERIMENTS_DIR, STATIC_DIR, copy_file, display_name, read_text


def build_space(
    experiment: str,
    run: Path,
    destination: Path,
    dataset_repo: str | None,
    include_checkpoint: bool,
) -> None:

    experiment_dir = EXPERIMENTS_DIR / experiment

    custom_readme = clean_experiment_readme(
        read_text(
            experiment_dir / "README.md"
        )
    )

    metrics = read_metrics(
        run / "metrics.csv"
    )

    samples = find_sample_images(
        run
    )

    results = load_results(run)

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------------------------------------------
    # Space README
    # ------------------------------------------------------------------

    space_readme = generate_space_readme(
        experiment=experiment,
        results=results,
        dataset_repo=dataset_repo,
    )

    (
        destination / "README.md"
    ).write_text(
        space_readme,
        encoding="utf-8",
    )


    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    data_dir = destination / "data"

    data_dir.mkdir(
        exist_ok=True
    )

    metrics_payload = generate_metrics_json(
            rows=metrics,
            samples=samples,
            experiment=experiment,
            run=run,
            results=results,
        )

    (
        data_dir / "metrics.json"
    ).write_text(
        json.dumps(
            metrics_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    # ------------------------------------------------------------------
    # Copy results.yaml
    # ------------------------------------------------------------------

    results_file = find_results_file(run)

    if results_file:
        copy_file(
            results_file,
            destination / results_file.name,
        )

    else:

        print(
            "WARNING: No results.yaml/results.yml "
            "found in run."
        )


    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    images_dir = destination / "images"

    plots = copy_plots(
        run,
        images_dir,
    )


    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    if include_checkpoint:

        checkpoint = find_best_checkpoint(run)

        if checkpoint:

            copy_file(
                checkpoint,
                destination /
                "checkpoints" /
                checkpoint.name,
            )

            print(
                f"Selected checkpoint: "
                f"{checkpoint.name}"
            )


    # ------------------------------------------------------------------
    # Interactive HTML
    # ------------------------------------------------------------------

    index = generate_index_html(
        experiment=experiment,
        custom_readme=custom_readme,
        results=results,
        metrics=metrics,
        samples=samples,
        plots=plots,
    )

    (
        destination / "index.html"
    ).write_text(
        index,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Static webpage elements
    # ------------------------------------------------------------------

    for f in STATIC_DIR.glob("*"):
        copy_file(f, destination)


    print(
        f"Built Space: {destination}"
    )

    print(
        f"Run:         {run}"
    )

    print(
        f"Samples:     {len(samples)}"
    )

    print(
        f"Metrics:     {len(metrics)}"
    )

    print(
        f"Plots:       {len(plots)}"
    )

    if results:
        print(
            "Results:     results.yaml"
        )
    else:
        print(
            "Results:     NOT FOUND"
        )


def generate_metrics_json(
    rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    experiment: str,
    run: Path,
    results: dict[str, Any],
) -> dict[str, Any]:

    return {
        "experiment": experiment,
        "display_name": display_name(experiment),
        "run": str(run.relative_to(ROOT)),
        "metrics": rows,
        "samples": samples,
        "results": results,
    }


def clean_experiment_readme(text: str) -> str:
    """
    Remove only the first H1.

    The generated page owns the main title.
    """

    if not text:
        return ""

    lines = text.splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)

    if lines and re.match(r"^#\s+", lines[0]):
        lines.pop(0)

    while lines and not lines[0].strip():
        lines.pop(0)

    return "\n".join(lines).strip()
