import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ufs2arco.targets.anemoi import Anemoi
from ufs2arco.utils import expand_anemoi_dataset


def _target() -> Anemoi:
    source = SimpleNamespace(
        time=pd.date_range("2017-01-01", periods=1, freq="6h"),
        fhr=None,
        member=None,
        sample_dims=("time",),
        horizontal_dims=("latitude", "longitude"),
    )
    return Anemoi(
        source=source,
        chunks={"time": 1, "variable": -1, "ensemble": 1, "cell": -1},
        store_path="unused.zarr",
        sort_channels_by_levels=True,
        use_level_index=False,
    )


def _pressure_dataset(target: Anemoi) -> xr.Dataset:
    levels = np.array([850, 200, 250])
    values = np.stack(
        [np.full((2, 2), level, dtype=np.float64) for level in levels],
        axis=0,
    )[None, ...]
    values[0, 0, 0, 0] = np.nan

    xds = xr.Dataset(
        {
            "ugrd_pl": (
                ("time", "level", "latitude", "longitude"),
                values,
            ),
        },
        coords={
            "time": target.source.time,
            "level": levels,
            "latitude": [-30.0, 30.0],
            "longitude": [0.0, 90.0],
        },
    )
    xds["level"].attrs.update(
        {
            "units": "hPa",
            "standard_name": "air_pressure",
            "type_of_level": "isobaricInhPa",
        }
    )
    xds["ugrd_pl"].attrs.update(
        {
            "units": "m s-1",
            "source_variable": "ugrd",
            "vertical_interpolation": "linear_in_log_pressure",
            "source_vertical_grid": "ufs_replay_127_layer_hybrid",
            "support_variables": ["pressfc", "tmp", "spfh", "delz"],
            "required_support_fields": np.array(["pressfc", "tmp", "spfh", "delz"]),
            "derived_from": ["pressfc", "tmp", "spfh", "delz", "ugrd"],
            "below_ground_policy": "mask",
            "unbracketed_policy": "bounded_recovery",
            "extrapolation_policy": (
                "linear_in_log_pressure_from_lower_pressure_pair_within_10_hpa"
            ),
            "nearest_neighbor_policy": (
                "single_lower_pressure_level_within_10_hpa"
            ),
            "max_fallback_distance_hpa": 10.0,
            "max_bridge_span_hpa": 25.0,
            "max_bridged_missing_levels": 1,
        }
    )
    return xds


def _packed_branch(names: list[str], start_value: float = 1.0) -> xr.Dataset:
    nvars = len(names)
    coords = {
        "time": [0],
        "variable": np.arange(nvars),
        "ensemble": [0],
        "cell": np.arange(4),
    }
    channel_values = start_value + np.arange(nvars, dtype=np.float32)
    data = np.broadcast_to(channel_values[None, :, None, None], (1, nvars, 1, 4)).copy()

    xds = xr.Dataset(
        {
            "data": xr.DataArray(
                data,
                dims=("time", "variable", "ensemble", "cell"),
                coords=coords,
            ),
            "count_array": xr.DataArray(
                np.full((1, nvars, 1), 4.0),
                dims=("time", "variable", "ensemble"),
                coords={key: coords[key] for key in ("time", "variable", "ensemble")},
            ),
            "has_nans_array": xr.DataArray(
                np.zeros((1, nvars, 1), dtype=bool),
                dims=("time", "variable", "ensemble"),
                coords={key: coords[key] for key in ("time", "variable", "ensemble")},
            ),
            "maximum_array": xr.DataArray(
                channel_values[None, :, None],
                dims=("time", "variable", "ensemble"),
                coords={key: coords[key] for key in ("time", "variable", "ensemble")},
            ),
            "minimum_array": xr.DataArray(
                channel_values[None, :, None],
                dims=("time", "variable", "ensemble"),
                coords={key: coords[key] for key in ("time", "variable", "ensemble")},
            ),
            "squares_array": xr.DataArray(
                (4.0 * channel_values**2)[None, :, None],
                dims=("time", "variable", "ensemble"),
                coords={key: coords[key] for key in ("time", "variable", "ensemble")},
            ),
            "sums_array": xr.DataArray(
                (4.0 * channel_values)[None, :, None],
                dims=("time", "variable", "ensemble"),
                coords={key: coords[key] for key in ("time", "variable", "ensemble")},
            ),
            "latitudes": xr.DataArray(
                [-30.0, -30.0, 30.0, 30.0],
                dims="cell",
                coords={"cell": coords["cell"]},
            ),
            "longitudes": xr.DataArray(
                [0.0, 90.0, 0.0, 90.0],
                dims="cell",
                coords={"cell": coords["cell"]},
            ),
        }
    )
    xds.attrs.update(
        {
            "variables": names,
            "variables_metadata": {
                name: {"test_channel": name}
                for name in names
            },
        }
    )
    return xds


def test_pressure_levels_pack_with_numeric_order_and_durable_metadata():
    target = _target()
    source = _pressure_dataset(target)

    packed = target.apply_transforms_to_sample(source)

    expected_names = ["ugrd_pl_200", "ugrd_pl_250", "ugrd_pl_850"]
    assert packed.attrs["variables"] == expected_names
    assert packed["data"].dtype == np.float32

    for pressure_hpa in (200, 250, 850):
        channel_name = f"ugrd_pl_{pressure_hpa}"
        channel_idx = expected_names.index(channel_name)
        expected = source["ugrd_pl"].sel(level=pressure_hpa).values.reshape(-1)
        actual = packed["data"].isel(
            time=0,
            variable=channel_idx,
            ensemble=0,
        ).values
        np.testing.assert_allclose(actual, expected, equal_nan=True)

        metadata = packed.attrs["variables_metadata"][channel_name]
        assert metadata["units"] == "m s-1"
        assert metadata["type_of_level"] == "isobaricInhPa"
        assert metadata["pressure_level_hpa"] == pressure_hpa
        assert metadata["level_units"] == "hPa"
        assert metadata["source_variable"] == "ugrd"
        assert metadata["vertical_interpolation"] == "linear_in_log_pressure"
        assert metadata["source_vertical_grid"] == "ufs_replay_127_layer_hybrid"
        assert metadata["support_variables"] == ["pressfc", "tmp", "spfh", "delz"]
        assert metadata["required_support_fields"] == ["pressfc", "tmp", "spfh", "delz"]
        assert metadata["derived_from"] == ["pressfc", "tmp", "spfh", "delz", "ugrd"]
        assert metadata["below_ground_policy"] == "mask"
        assert metadata["unbracketed_policy"] == "bounded_recovery"
        assert (
            metadata["extrapolation_policy"]
            == "linear_in_log_pressure_from_lower_pressure_pair_within_10_hpa"
        )
        assert (
            metadata["nearest_neighbor_policy"]
            == "single_lower_pressure_level_within_10_hpa"
        )
        assert metadata["max_fallback_distance_hpa"] == 10.0
        assert metadata["max_bridge_span_hpa"] == 25.0
        assert metadata["max_bridged_missing_levels"] == 1
        assert metadata["mars"]["param"] == "ugrd_pl"
        assert metadata["mars"]["variable"] == "ugrd_pl"
        assert metadata["mars"]["level"] == pressure_hpa
        assert metadata["mars"]["levelist"] == pressure_hpa
        assert metadata["mars"]["levtype"] == "pl"
        json.dumps(metadata)

    index_850 = expected_names.index("ugrd_pl_850")
    assert packed["count_array"].isel(variable=index_850).item() == 3
    assert packed["has_nans_array"].isel(variable=index_850).item()
    for stat_name in target.packed_stat_names:
        assert packed[stat_name].sizes["variable"] == len(expected_names)


def test_multisource_merge_appends_pressure_channels_and_keeps_stats_aligned():
    target = _target()
    base_names = ["tmp", "ugrd_25", "ugrd_850", "z_static"]
    pressure_names = ["ugrd_pl_200", "ugrd_pl_250", "ugrd_pl_850"]
    base = _packed_branch(base_names, start_value=10.0)
    pressure = _packed_branch(pressure_names, start_value=20.0)

    merged = target.merge_multisource([base, pressure])

    expected_names = base_names + pressure_names
    assert merged.attrs["variables"] == expected_names
    assert list(merged.attrs["variables_metadata"]) == expected_names
    np.testing.assert_array_equal(merged["variable"], np.arange(len(expected_names)))
    assert "variable" not in merged["latitudes"].dims
    assert "variable" not in merged["longitudes"].dims

    expected_values = [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0]
    np.testing.assert_allclose(
        merged["data"].isel(time=0, ensemble=0, cell=0),
        expected_values,
    )
    np.testing.assert_allclose(
        merged["sums_array"].isel(time=0, ensemble=0),
        4.0 * np.asarray(expected_values),
    )
    for stat_name in target.packed_stat_names:
        assert merged[stat_name].sizes["variable"] == len(expected_names)


def test_expansion_keeps_finite_volume_and_pressure_winds_separate():
    target = _target()
    base = _packed_branch(["tmp", "ugrd_25", "ugrd_850"], start_value=10.0)
    pressure = _packed_branch(
        ["ugrd_pl_200", "ugrd_pl_250", "ugrd_pl_850"],
        start_value=20.0,
    )
    merged = target.merge_multisource([base, pressure])
    merged["dates"] = xr.DataArray(
        pd.date_range("2017-01-01", periods=1, freq="6h"),
        dims="time",
    )

    expanded = expand_anemoi_dataset(
        merged,
        "data",
        merged.attrs["variables"],
    )

    assert "ugrd" in expanded
    assert "ugrd_pl" in expanded
    coarse_wind = expanded["ugrd"].dropna("level", how="all")
    pressure_wind = expanded["ugrd_pl"].dropna("level", how="all")
    np.testing.assert_array_equal(coarse_wind["level"], [25, 850])
    np.testing.assert_array_equal(pressure_wind["level"], [200, 250, 850])
    np.testing.assert_allclose(expanded["ugrd"].sel(level=25), 11.0)
    np.testing.assert_allclose(expanded["ugrd"].sel(level=850), 12.0)
    np.testing.assert_allclose(expanded["ugrd_pl"].sel(level=200), 20.0)
    np.testing.assert_allclose(expanded["ugrd_pl"].sel(level=250), 21.0)
    np.testing.assert_allclose(expanded["ugrd_pl"].sel(level=850), 22.0)


def test_multisource_merge_rejects_duplicate_channel_names():
    target = _target()
    first = _packed_branch(["tmp", "ugrd_pl_850"])
    second = _packed_branch(["ugrd_pl_850"])

    with pytest.raises(ValueError, match="Duplicate packed channel names"):
        target.merge_multisource([first, second])


@pytest.mark.parametrize("field", ["cell", "latitudes", "longitudes"])
def test_multisource_merge_rejects_nonidentical_packed_grids(field):
    target = _target()
    first = _packed_branch(["tmp"])
    second = _packed_branch(["ugrd_pl_850"])

    if field == "cell":
        second = second.assign_coords(cell=second["cell"].values + 1)
    else:
        second[field] = second[field].copy()
        second[field].values[0] += 0.25

    with pytest.raises(ValueError, match=f"packed grid field '{field}'"):
        target.merge_multisource([first, second])
