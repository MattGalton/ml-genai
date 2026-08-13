from pathlib import Path
import re
import shutil
from typing import Any
import yaml

FILE_DIR = Path(__file__).resolve().parent
ROOT = FILE_DIR.parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
TEMPLATES_DIR = FILE_DIR / "templates"
STATIC_DIR = FILE_DIR / "static"
PUBLISH_DIR = ROOT / ".hf_publish"


def display_name(experiment: str) -> str:
    """
    03_MADE -> MADE
    01_FVSBN -> FVSBN
    """
    return re.sub(r"^\d+_", "", experiment).replace("_", " ")


def die(message: str) -> None:
    raise SystemExit(f"\nERROR: {message}\n")


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return {}

    if not isinstance(data, dict):
        die(f"Expected YAML object in {path}")

    return data

def read_text(path: Path) -> str:
    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8").strip()
