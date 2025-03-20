import pytest
import os
from unittest.mock import patch, MagicMock
import xarray as xr
import pandas as pd
from ufs2arco.datamover import DataMover

@pytest.fixture
def source():
    source = MagicMock()
    source.open_sample_dataset = MagicMock(return_value=xr.Dataset())
    source.name = "MockDataset"
    source.t0 = pd.date_range(start="2022-01-01T00", end="2022-01-01T18", freq="6h")
    source.fhr = [0, 6]
    source.member = [0, 1]
    source.sample_dims = ("t0", "fhr", "member")
    return source

@pytest.fixture
def target():
    target = MagicMock()
    target.chunks = {"t0": 1, "fhr": 1, "member": 1}
    target.store_path = "/tmp/store/gefsdataset.zarr"
    source.apply_transforms_to_sample = MagicMock(return_value=xr.Dataset())
    return target


@pytest.fixture
def data_mover(source, target):
    return DataMover(source, target, batch_size=2, start=0, cache_dir=".")

def test_init(data_mover):
    assert data_mover.batch_size == 2
    assert data_mover.counter == 0
    assert data_mover.data_counter == 0
    assert data_mover.outer_cache_dir == "."
    assert len(data_mover.sample_indices) == 16

def test_len(data_mover):
    assert len(data_mover) == 8

def test_get_cache_dir(data_mover):
    assert data_mover.get_cache_dir(0) == "./datamover-cache/0"

def test_get_batch_indices(data_mover):
    batch_indices = data_mover.get_batch_indices(0)
    assert len(batch_indices) == 2

@patch("ufs2arco.datamover.xr.merge")
def test__next_data(mock_merge, data_mover):
    mock_merge.return_value = xr.Dataset()
    result = data_mover._next_data()
    assert isinstance(result, xr.Dataset)
    mock_merge.assert_called_once()

@patch("ufs2arco.datamover.xr.merge")
def test_get_data(mock_merge, data_mover):
    mock_merge.return_value = xr.Dataset()
    result = data_mover.get_data()
    assert isinstance(result, xr.Dataset)

@patch("ufs2arco.datamover.xr.merge")
def test_clear_cache(mock_merge, data_mover):
    mock_merge.return_value = xr.Dataset()
    next(data_mover)
    data_mover.clear_cache(0)
    assert not os.path.isdir("./datamover-cache/0")

def test_restart(data_mover):
    data_mover.restart()
    assert data_mover.data_counter == 0
