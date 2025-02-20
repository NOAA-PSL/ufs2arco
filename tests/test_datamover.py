import pytest
import os
from unittest.mock import patch, MagicMock
import xarray as xr
import pandas as pd
from ufs2arco.datamover import DataMover

@pytest.fixture
def dataset():
    dataset = MagicMock()
    dataset.chunks = {"t0": 1, "fhr": 1, "member": 1}
    dataset.open_single_dataset = MagicMock(return_value=xr.Dataset())
    dataset.name = "MockDataset"
    dataset.t0 = pd.date_range(start="2022-01-01T00", end="2022-01-01T18", freq="6h")
    dataset.fhr = [0, 6]
    dataset.member = [0, 1]
    return dataset

@pytest.fixture
def data_mover(dataset):
    return DataMover(dataset, batch_size=2, sample_dims=["t0", "fhr", "member"], start=0, cache_dir=".")

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

def test_get_data(data_mover):
    result = data_mover.get_data()
    assert isinstance(result, xr.Dataset)

def test_clear_cache(data_mover):
    next(data_mover)
    data_mover.clear_cache(0)
    assert not os.path.isdir("./datamover-cache/0")

def test_restart(data_mover):
    data_mover.restart()
    assert data_mover.data_counter == 0
