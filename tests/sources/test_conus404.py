import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ufs2arco.sources.conus404 import CONUS404
from ufs2arco.transforms.destagger import destagger


TIME = {"start": "2000-01-01", "end": "2000-01-02", "freq": "1h"}
URI = "abfs://test/conus404.zarr"


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

NZ = 50

def _wrf_store() -> xr.Dataset:
    """Native CONUS404 layout: WRF dimension names and no vertical coordinates."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2017-01-01T00", periods=2, freq="1h")
    return xr.Dataset(
        {
            "TK": (("time", "bottom_top", "south_north", "west_east"), rng.normal(size=(2, NZ, 2, 3))),
            "U": (("time", "bottom_top", "south_north", "west_east_stag"), rng.normal(size=(2, NZ, 2, 4))),
            "V": (("time", "bottom_top", "south_north_stag", "west_east"), rng.normal(size=(2, NZ, 3, 3))),
            "Z": (
                ("time", "bottom_top_stag", "south_north", "west_east"),
                np.cumsum(rng.uniform(20, 500, (2, NZ + 1, 2, 3)), axis=1),
            ),
            "T2": (("time", "south_north", "west_east"), rng.normal(size=(2, 2, 3))),
        },
        coords={"time": time},
    )


@pytest.fixture
def opened(monkeypatch):
    calls = []
    store = _wrf_store()

    def open_zarr(uri, **kwargs):
        calls.append(kwargs)
        return store

    monkeypatch.setattr(xr, "open_zarr", open_zarr)
    return store, calls


def _conus404(**kwargs) -> CONUS404:
    return CONUS404(
        time={"start": "2017-01-01T00", "end": "2017-01-01T01", "freq": "1h"},
        uri=URI,
        storage_options={"account_name": "test"},
        **kwargs,
    )


def _z_layers(z, levels):
    return 0.5 * (z[:, :-1] + z[:, 1:])[:, levels]


def test_open_passes_storage_options_and_labels_native_levels(opened):
    _, calls = opened
    source = _conus404(variables=["TK", "Z"])

    assert calls[0]["storage_options"] == {"account_name": "test"}
    np.testing.assert_array_equal(source._xds["level"], np.arange(NZ))
    np.testing.assert_array_equal(source._xds["level_stag"], np.arange(NZ + 1))


def test_noncontiguous_levels_destagger_to_selected_layers(opened):
    store, _ = opened
    levels = [0, 10, 25]
    source = _conus404(variables=["TK", "Z", "U", "V", "T2"], levels=levels)

    np.testing.assert_array_equal(source._xds["level"], levels)
    np.testing.assert_array_equal(source._xds["level_stag"], [0, 1, 10, 11, 25, 26])

    result = destagger(source._xds, {"Z": "level", "U": "longitude", "V": "latitude"})

    for name in ("Z", "U", "V"):
        assert result[name].dims == ("time", "level", "latitude", "longitude")
    np.testing.assert_allclose(result["Z"].values, _z_layers(store["Z"].values, levels))
    np.testing.assert_array_equal(result["Z"]["level"], levels)
    assert not {"level_stag", "latitude_stag", "longitude_stag"} & set(result.dims)


def test_staggered_only_sample_keeps_level_labels(opened):
    store, _ = opened
    levels = [0, 10, 25]
    source = _conus404(variables=["Z"], levels=levels)
    sample = source.open_sample_dataset(
        dims={"time": pd.Timestamp("2017-01-01T00")},
        open_static_vars=False,
    )
    result = destagger(sample, {"Z": "level"})

    np.testing.assert_array_equal(result["level"], levels)
    np.testing.assert_allclose(result["Z"].values, _z_layers(store["Z"].values[[0]], levels))


def test_nearest_levels_label_staggered_selection(opened):
    store, _ = opened
    source = _conus404(variables=["TK", "Z"], levels=[4.6, 9.8], use_nearest_levels=True)
    result = destagger(source._xds, {"Z": "level"})

    np.testing.assert_array_equal(result["level"], [5, 10])
    np.testing.assert_allclose(result["Z"].values, _z_layers(store["Z"].values, [5, 10]))
