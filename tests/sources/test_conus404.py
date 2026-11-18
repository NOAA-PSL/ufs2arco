import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ufs2arco.sources.conus404 import CONUS404


TIME = {"start": "2000-01-01", "end": "2000-01-02", "freq": "1h"}
URI = "abfs://test/conus404.zarr"


def test_class_attributes():
    assert CONUS404.sample_dims == ("time",)
    assert CONUS404.horizontal_dims == ("latitude", "longitude")


def test_get_storage_options_reads_signed_zarr_asset():
    storage_options = {"account_name": "test"}
    pystac_client = MagicMock()
    collection = pystac_client.Client.open.return_value.get_collection.return_value
    collection.assets = {
        "zarr-abfs": MagicMock(extra_fields={"xarray:storage_options": storage_options})
    }
    planetary_computer = MagicMock()

    with patch.dict(
        sys.modules,
        {"pystac_client": pystac_client, "planetary_computer": planetary_computer},
    ):
        assert CONUS404.get_storage_options() == storage_options

    assert pystac_client.Client.open.call_args.kwargs["modifier"] is planetary_computer.sign_inplace
    pystac_client.Client.open.return_value.get_collection.assert_called_once_with("conus404")


def test_available_levels():
    obj = CONUS404.__new__(CONUS404)
    assert obj.available_levels == tuple(range(50))


def test_available_stag_levels():
    obj = CONUS404.__new__(CONUS404)
    assert obj.available_stag_levels == tuple(range(51))


def test_rename_mapping():
    obj = CONUS404.__new__(CONUS404)
    assert obj.rename == {
        "bottom_top": "level",
        "bottom_top_stag": "level_stag",
        "south_north": "latitude",
        "west_east": "longitude",
        "south_north_stag": "latitude_stag",
        "west_east_stag": "longitude_stag",
    }


@pytest.mark.parametrize("explicit", [None, {"account_name": "explicit"}])
def test_init_sets_time_and_storage_options(explicit):
    fetched = {"account_name": "fetched"}
    with patch.object(CONUS404, "get_storage_options", return_value=fetched) as fetch, patch(
        "ufs2arco.sources.conus404.CloudZarrData.__init__", return_value=None
    ) as super_init:
        obj = CONUS404(time=TIME, uri=URI, storage_options=explicit)

    assert obj.time.equals(pd.date_range(**TIME))
    assert obj.storage_options == (explicit or fetched)
    assert fetch.call_count == (0 if explicit else 1)
    assert super_init.call_args.kwargs["uri"] == URI
