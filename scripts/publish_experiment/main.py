import argparse
import re
import os
import shutil

from .build import build_space
from .huggingface import upload_space
from .runs import find_run
from .vars import EXPERIMENTS_DIR, PUBLISH_DIR, die


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Publish an ML experiment as an "
            "interactive Hugging Face Space."
        )
    )

    parser.add_argument(
        "experiment",
        help=(
            "Experiment directory, "
            "e.g. 03_MADE"
        ),
    )

    parser.add_argument(
        "--run",
        help=(
            "Run relative to outputs/, e.g. "
            "MADE_BMNIST/2026-08-09/23-34-46"
        ),
    )

    parser.add_argument(
        "--repo-id",
        help=(
            "Hugging Face Space ID, e.g. "
            "MattGalton/ml-genai-made"
        ),
    )

    parser.add_argument(
        "--hf-username",
        default=os.environ.get(
            "HF_USERNAME"
        ),
        help="Hugging Face username.",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private.",
    )

    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help=(
            "Publish the selected best checkpoint."
        ),
    )

    parser.add_argument(
        "--dataset-repo",
        help=(
            "Optional HF dataset repo to link "
            "from the Space metadata."
        ),
    )

    parser.add_argument(
        "--push",
        action="store_true",
        help=(
            "Upload the generated Space "
            "to Hugging Face."
        ),
    )

    args = parser.parse_args()


    experiment = args.experiment

    experiment_dir = EXPERIMENTS_DIR / experiment

    if not experiment_dir.is_dir():
        die(
            f"Experiment does not exist: "
            f"{experiment_dir}"
        )


    run = find_run(
        experiment_dir,
        args.run,
    )


    destination = PUBLISH_DIR / experiment


    if destination.exists():
        shutil.rmtree(destination)


    build_space(
        experiment=experiment,
        run=run,
        destination=destination,
        dataset_repo=args.dataset_repo,
        include_checkpoint=args.checkpoint,
    )


    if not args.push:

        print()
        print("Dry run complete.")
        print()

        print("Open locally with:")
        print()

        print(
            f"  open "
            f"{destination / 'index.html'}"
        )

        print()
        print(
            "Add --push when you're happy with it."
        )

        return


    repo_id = args.repo_id


    if not repo_id:

        if not args.hf_username:

            die(
                "Provide --repo-id "
                "or set HF_USERNAME."
            )

        repo_id = (
            f"{args.hf_username}/"
            f"ml-genai-{slugify(experiment)}"
        )


    upload_space(
        local_dir=destination,
        repo_id=repo_id,
        private=args.private,
    )