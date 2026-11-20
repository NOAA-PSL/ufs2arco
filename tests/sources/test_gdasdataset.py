import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import xarray as xr

from ufs2arco.sources import GDASArchive


@pytest.fixture
def gdas_dataset():
    t0 = {"start": "2025-01-01T00", "end": "2025-01-01T18", "freq": "6h"}
    fhr = {"start": 0, "end": 6, "step": 6}
    resolution = "1p00"
    chunks = {"t0": 1, "fhr": 1, "latitude": -1, "longitude": -1}
    return GDASArchive(
        t0, fhr, resolution, variables=["t", "t2m", "prmsl"], levels=[100, 500, 1000]
    )


def test_init(gdas_dataset):
    assert len(gdas_dataset.t0) == 4
    assert np.array_equal(gdas_dataset.fhr, np.array([0, 6]))


def test_str(gdas_dataset):
    str(gdas_dataset)  # just make sure this can run without bugs


def test_name(gdas_dataset):
    assert gdas_dataset.name == "GDASArchive"


def test_open_sample_dataset(gdas_dataset):
    result = gdas_dataset.open_sample_dataset(
        dims={
            "t0": pd.Timestamp("2025-01-01T00"),
            "fhr": 0,
        },
        open_static_vars=True,
        cache_dir="/tmp/cache",
    )
    assert isinstance(result, xr.Dataset)
