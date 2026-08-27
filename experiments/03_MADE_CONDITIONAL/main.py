from pathlib import Path

from mattg.training.train import train_binary_autoregressive

FILE_DIR = Path(__file__).parent

if __name__ == "__main__":
    train_binary_autoregressive(hydra_config_path=FILE_DIR)