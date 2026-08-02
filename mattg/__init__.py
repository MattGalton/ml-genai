from pathlib import Path

HOME = Path.home() / ".mattg" / "data"
HOME.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path(__file__).parent / "config" / "cfg"

if not CONFIG_PATH.is_dir():
    raise FileNotFoundError(
        f"Could not find Hydra config directory at {CONFIG_PATH}"
    )

__all__ = ["HOME", "CONFIG_PATH"]
