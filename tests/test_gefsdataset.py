import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import xarray as xr

from ufs2arco.gefsdataset import GEFSDataset

@pytest.fixture
def gefs_dataset():
    t0 = {"start": "2017-01-01T00", "end": "2017-01-01T18", "freq": "6h"}
    fhr = {"start": 0, "end": 6}
    member = {"start": 0, "end": 1}
    chunks = {"t0": 1, "fhr": 1, "member": 1, "latitude": -1, "longitude": -1}
    store_path = "/tmp/store/gefsdataset.zarr"
    return GEFSDataset(t0, fhr, member, chunks, store_path)

def test_init(gefs_dataset):
    assert len(gefs_dataset.t0) == 4
    assert np.array_equal(gefs_dataset.fhr, np.array([0, 6]))
    assert np.array_equal(gefs_dataset.member, np.array([0, 1]))
    assert gefs_dataset.store_path == "/tmp/store/gefsdataset.zarr"
    assert gefs_dataset.chunks == {"t0": 1, "fhr": 1, "member": 1, "latitude": -1, "longitude": -1}

def test_len(gefs_dataset):
    assert len(gefs_dataset) == 4

def test_str(gefs_dataset):
    str(gefs_dataset) # just make sure this can run without bugs

def test_name(gefs_dataset):
    assert gefs_dataset.name == "GEFSDataset"

@patch("ufs2arco.gefsdataset.xr.Dataset.to_zarr")
@patch("ufs2arco.gefsdataset.GEFSDataset.open_single_dataset")
def test_create_container(mock_open_single_dataset, mock_to_zarr, gefs_dataset):
    mock_open_single_dataset.return_value = MagicMock()
    gefs_dataset.create_container()
    mock_to_zarr.assert_called_once_with(gefs_dataset.store_path, compute=False)

@patch("ufs2arco.gefsdataset.xr.merge")
@patch("ufs2arco.gefsdataset.GEFSDataset.open_single_initial_condition")
def test_open_dataset(mock_open_single_initial_condition, mock_merge, gefs_dataset):
    mock_open_single_initial_condition.return_value = MagicMock()
    mock_merge.return_value = MagicMock()
    result = gefs_dataset.open_dataset()
    assert mock_open_single_initial_condition.call_count == len(gefs_dataset.t0)
    mock_merge.assert_called()
    assert result == mock_merge.return_value

def test_open_single_dataset(gefs_dataset):
    result = gefs_dataset.open_single_dataset(pd.Timestamp("2017-01-01T00"), 0, 0, "/tmp/cache")
    assert isinstance(result, xr.Dataset)
