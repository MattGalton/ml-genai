from scripts.publish_experiment.metrics import read_metrics
from scripts.publish_experiment.runs import find_sample_images


def test_read_metrics_converts_zero_based_epochs_for_display(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(
        "epoch,train_loss\n"
        "0,0.5\n"
        "9,0.1\n",
        encoding="utf-8",
    )

    assert read_metrics(metrics_path) == [
        {
            "epoch": 1.0,
            "train_loss": 0.5,
        },
        {
            "epoch": 10.0,
            "train_loss": 0.1,
        },
    ]


def test_find_sample_images_converts_zero_based_epochs_for_display(tmp_path):
    plots_path = tmp_path / "plots"
    plots_path.mkdir()
    (plots_path / "samples_epoch_0000.png").write_bytes(b"")
    (plots_path / "samples_epoch_0009.png").write_bytes(b"")

    assert find_sample_images(tmp_path) == [
        {
            "epoch": 1,
            "metric_epoch": 0,
            "filename": "samples_epoch_0000.png",
        },
        {
            "epoch": 10,
            "metric_epoch": 9,
            "filename": "samples_epoch_0009.png",
        },
    ]
