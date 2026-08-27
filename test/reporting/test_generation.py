from types import SimpleNamespace

import torch
import yaml
from omegaconf import OmegaConf

from mattg.reporting.generation import build_results, config_summary, to_python


def test_to_python_converts_omegaconf_containers_for_yaml():
    value = {
        "shape": OmegaConf.create([1, 28, 28]),
        "nested": OmegaConf.create({"name": "Binarized MNIST"}),
    }

    converted = to_python(value)

    assert converted == {
        "shape": [1, 28, 28],
        "nested": {
            "name": "Binarized MNIST",
        },
    }
    yaml.safe_dump(converted)


def test_config_summary_omits_framework_wiring():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "data_spec": {
                    "_target_": "mattg.datasets.DataSpec",
                    "data_dim": 784,
                    "num_classes": 10,
                    "name": "Binarized MNIST",
                    "shape": [1, 28, 28],
                },
                "train": {
                    "name": "BMNIST",
                    "loader": {
                        "_target_": "torch.utils.data.DataLoader",
                        "batch_size": 32,
                    },
                },
                "val": {
                    "name": "BMNIST",
                    "loader": {
                        "_target_": "torch.utils.data.DataLoader",
                        "batch_size": 32,
                    },
                },
            },
            "optimizer": {
                "name": "Adam",
                "target": {
                    "_target_": "torch.optim.Adam",
                    "lr": 0.001,
                },
            },
            "trainer": {
                "_target_": "lightning.Trainer",
                "max_epochs": 50,
                "precision": 32,
                "callbacks": [
                    {
                        "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "save_top_k": 3,
                    },
                ],
            },
            "model": {
                "name": "FVSBN",
                "target": {
                    "_target_": "mattg.models.FVSBNFactory",
                    "in_features": 784,
                    "out_features": 784,
                },
            },
        }
    )

    assert config_summary(cfg) == {
        "dataset": {
            "data_dim": 784,
            "num_classes": 10,
            "name": "Binarized MNIST",
            "shape": [1, 28, 28],
        },
        "optimizer": {
            "name": "Adam",
            "lr": 0.001,
        },
        "trainer": {
            "max_epochs": 50,
            "precision": 32,
        },
        "model": {
            "name": "FVSBN",
            "in_features": 784,
            "out_features": 784,
        },
    }


def test_build_results_reports_completed_epochs_without_extra_increment():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "data_spec": {
                    "name": "Binarized MNIST",
                    "data_dim": 784,
                },
            },
            "model": {
                "name": "FVSBN",
                "target": {
                    "in_features": 784,
                },
            },
        }
    )
    trainer = SimpleNamespace(
        current_epoch=10,
        global_step=100,
        callback_metrics={},
        callbacks=[],
    )
    pl_module = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        data_spec=None,
    )

    results = build_results(trainer, pl_module, cfg)

    assert results["training"]["epochs"] == 10
    assert "dataset" not in results
    assert "model" not in results
    assert results["config"]["dataset"]["name"] == "Binarized MNIST"
    assert results["config"]["model"]["name"] == "FVSBN"
