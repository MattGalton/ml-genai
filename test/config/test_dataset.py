import pytest
from omegaconf import OmegaConf

from mattg.config.dataset import (
    DatasetLoader,
    DatasetConfig,
    SingleDatasetConfig,
    DatasetImplConfig,
    DataLoaderConfig,
)


class DummyDataset:
    def __init__(self, train):
        self.train = train


class DummyLoader:
    def __init__(self, dataset, batch_size, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers


@pytest.fixture
def train_cfg_raw():
    return SingleDatasetConfig(
        dataset=DatasetImplConfig(
            _target_=f"{__name__}.DummyDataset",
        ),
        loader=DataLoaderConfig(
            _target_=f"{__name__}.DummyLoader",
            batch_size=32,
            num_workers=4,
        ),
    )


@pytest.fixture
def val_cfg_raw():
    return SingleDatasetConfig(
        name="dummy",
        dataset=DatasetImplConfig(
            _target_=f"{__name__}.DummyDataset",
        ),
        loader=DataLoaderConfig(
            _target_=f"{__name__}.DummyLoader",
            batch_size=16,
            num_workers=2,
        ),
    )


@pytest.fixture
def train_cfg(train_cfg_raw):
    return OmegaConf.structured(train_cfg_raw)


@pytest.fixture
def val_cfg(val_cfg_raw):
    return OmegaConf.structured(val_cfg_raw)


def test_load_single_train(train_cfg):
    loader = DatasetLoader._load_single(train_cfg, train=True)

    assert isinstance(loader, DummyLoader)
    assert isinstance(loader.dataset, DummyDataset)
    assert loader.dataset.train is True
    assert loader.batch_size == 32
    assert loader.num_workers == 4


def test_load_single_val(train_cfg):
    loader = DatasetLoader._load_single(train_cfg, train=False)

    assert isinstance(loader, DummyLoader)
    assert isinstance(loader.dataset, DummyDataset)
    assert loader.dataset.train is False


def test_load_returns_train_and_val(train_cfg_raw, val_cfg_raw):
    cfg = OmegaConf.structured(DatasetConfig(
        train=train_cfg_raw,
        val=val_cfg_raw,
    ))

    train_loader, val_loader = DatasetLoader.load(cfg)

    assert isinstance(train_loader, DummyLoader)
    assert isinstance(val_loader, DummyLoader)

    assert train_loader.dataset.train is True
    assert val_loader.dataset.train is False

    assert train_loader.batch_size == 32
    assert train_loader.num_workers == 4

    assert val_loader.batch_size == 16
    assert val_loader.num_workers == 2


def test_load_without_validation(train_cfg):
    cfg = DatasetConfig(
        train=train_cfg,
        val=None,
    )

    train_loader, val_loader = DatasetLoader.load(cfg)

    assert isinstance(train_loader, DummyLoader)
    assert val_loader is None


@pytest.mark.parametrize(
    "loader_kwargs",
    [
        {"batch_size": 32},
        {"num_workers": 4},
        {},
    ],
)
def test_missing_required_dataloader_fields(loader_kwargs):
    cfg = OmegaConf.structured(
        SingleDatasetConfig(
            dataset=DatasetImplConfig(
                _target_=f"{__name__}.DummyDataset",
            ),
            loader=DataLoaderConfig(
                _target_=f"{__name__}.DummyLoader",
                **loader_kwargs,
            ),
        )
    )

    with pytest.raises(Exception):
        DatasetLoader._load_single(cfg, train=True)


def test_missing_dataset_target():
    cfg = OmegaConf.structured(
        SingleDatasetConfig(
            dataset=DatasetImplConfig(),
            loader=DataLoaderConfig(
                _target_=f"{__name__}.DummyLoader",
                batch_size=32,
                num_workers=4,
            ),
        )
    )

    with pytest.raises(Exception):
        DatasetLoader._load_single(cfg, train=True)