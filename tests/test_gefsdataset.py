import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import xarray as xr

from ufs2arco.sources import AWSGEFSArchive

@pytest.fixture
def gefs_dataset():
    t0 = {"start": "2017-01-01T00", "end": "2017-01-01T18", "freq": "6h"}
    fhr = {"start": 0, "end": 6, "step": 6}
    member = {"start": 0, "end": 1, "step": 1}
    chunks = {"t0": 1, "fhr": 1, "member": 1, "latitude": -1, "longitude": -1}
    return AWSGEFSArchive(t0, fhr, member, variables=["t", "t2m", "lsm"], levels=[100, 500, 1000])

def test_init(gefs_dataset):
    assert len(gefs_dataset.t0) == 4
    assert np.array_equal(gefs_dataset.fhr, np.array([0, 6]))
    assert np.array_equal(gefs_dataset.member, np.array([0, 1]))

def test_str(gefs_dataset):
    str(gefs_dataset) # just make sure this can run without bugs

def test_name(gefs_dataset):
    assert gefs_dataset.name == "AWSGEFSArchive"

def test_open_sample_dataset(gefs_dataset):
    result = gefs_dataset.open_sample_dataset(
        dims={
            "t0": pd.Timestamp("2017-01-01T00"),
            "fhr": 0,
            "member": 0,
        },
        open_static_vars=True,
        cache_dir="/tmp/cache",
    )
    assert isinstance(result, xr.Dataset)
