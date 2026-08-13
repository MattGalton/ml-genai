import csv
from pathlib import Path
import re
from typing import Any

def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        rows: list[dict[str, Any]] = []

        for row in reader:
            converted: dict[str, Any] = {}

            for key, value in row.items():
                if value is None or value == "":
                    converted[key] = value
                    continue

                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value

            rows.append(converted)

        return rows

def find_best_checkpoint(
    run: Path,
) -> Path | None:

    checkpoints = run / "checkpoints"

    if not checkpoints.exists():
        return None

    candidates = list(
        checkpoints.glob("*.ckpt")
    )

    if not candidates:
        return None

    scored: list[
        tuple[float, Path]
    ] = []

    for path in candidates:

        match = re.search(
            r"val_loss=([0-9.eE+-]+)",
            path.name,
        )

        if not match:
            continue

        try:
            score = float(
                match.group(1)
            )
        except ValueError:
            continue

        scored.append(
            (score, path)
        )

    if scored:
        return min(
            scored,
            key=lambda x: x[0],
        )[1]

    for path in candidates:
        if path.name == "last.ckpt":
            return path

    return candidates[0]