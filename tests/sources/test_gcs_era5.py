from unittest.mock import patch

import pandas as pd
import pytest

from ufs2arco.sources.gcs_era5 import GCSERA5

TIME = {"start": "2020-01-01T00", "end": "2020-01-01T06", "freq": "6h"}
DEFAULTS = {"variables": None, "levels": None, "use_nearest_levels": False, "slices": None}


def test_init_sets_time():
    with patch("ufs2arco.sources.gcs_era5.CloudZarrData.__init__", return_value=None):
        source = GCSERA5(time=TIME, uri="gcs://dummy.zarr")
    assert source.time.equals(pd.date_range(**TIME))


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"variables": ["t"], "levels": [500], "use_nearest_levels": True, "slices": {"time": slice(0, 1)}}],
)
def test_init_forwards_kwargs_to_super(kwargs):
    with patch("ufs2arco.sources.gcs_era5.CloudZarrData.__init__", return_value=None) as super_init:
        GCSERA5(time=TIME, uri="gcs://dummy.zarr", **kwargs)

    passed = super_init.call_args.kwargs
    assert passed["uri"] == "gcs://dummy.zarr"
    for key, value in {**DEFAULTS, **kwargs}.items():
        assert passed[key] == value
    assert "storage_options" not in passed
