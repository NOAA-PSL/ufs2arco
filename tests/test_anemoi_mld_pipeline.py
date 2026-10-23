from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.targets.anemoi import Anemoi
from ufs2arco.utils import expand_anemoi_dataset


LATITUDE = [-30.0, 30.0]
LONGITUDE = [0.0, 90.0]


def _target(multisource_masks=None) -> Anemoi:
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
        use_level_index=True,
        variables_with_nans=["mld"],
        multisource_masks=multisource_masks,
    )


def _mld_dataset(
    target: Anemoi,
    values: np.ndarray | None = None,
    uo_values: np.ndarray | None = None,
) -> xr.Dataset:
    if values is None:
        values = np.array([[np.nan, 10.0], [20.0, 30.0]], dtype=np.float64)

    data_vars = {
        "mld": (
            ("time", "latitude", "longitude"),
            np.asarray(values, dtype=np.float64)[None, ...],
            {
                "diagnostic": "mixed_layer_depth",
                "units": "m",
                "density_threshold": 0.03,
                "density_threshold_units": "kg m-3",
                "reference_pressure_pa": 0.0,
                "reference_density_layer": "top_model_layer",
                "equation_of_state": "wright_full",
                "source_temperature": "temp",
                "source_salinity": "so",
                "vertical_grid": "validated complete native grid",
                "vertical_coordinate": "nominal_layer_center_depth",
                "source_vertical_level_count": 75,
                "land_policy": "NaN over land",
                "bottom_policy": "deepest valid layer center",
                "missing_policy": "contiguous valid column",
                "pathological_policy": "NaN",
                "provenance": "MOM6 diagnoseMLDbyDensityDifference",
            },
        ),
    }
    coords = {
        "time": target.source.time,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
    }

    if uo_values is not None:
        data_vars["uo"] = (
            ("time", "level", "latitude", "longitude"),
            np.asarray(uo_values, dtype=np.float64)[None, None, ...],
        )
        coords["level"] = [0.5]

    return xr.Dataset(data_vars, coords=coords)


def _atmosphere_dataset(target: Anemoi) -> xr.Dataset:
    dims = ("time", "latitude", "longitude")
    coords = {
        "time": target.source.time,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
    }
    return xr.Dataset(
        {
            "pressfc": (dims, np.full((1, 2, 2), 100_000.0)),
            "tmp2m": (dims, np.full((1, 2, 2), 280.0)),
        },
        coords=coords,
    )


def _ohc_dataset(target: Anemoi) -> xr.Dataset:
    values = np.array([[np.nan, 12.0], [13.0, 14.0]], dtype=np.float64)
    return xr.Dataset(
        {
            "ohc300": (
                ("time", "latitude", "longitude"),
                values[None, ...],
                {
                    "diagnostic": "ocean_heat_content",
                    "units": "GJ m-2",
                    "native_units": "J m-2",
                    "joules_to_output_units": 1.0e-9,
                    "source_temperature": "temp",
                    "temperature_reference": 0.0,
                    "temperature_reference_units": "degC",
                    "integration_depth": 300.0,
                    "integration_depth_units": "m",
                    "rho0": 1035.0,
                    "rho0_units": "kg m-3",
                    "heat_capacity": 3991.86795711963,
                    "heat_capacity_units": "J kg-1 K-1",
                    "require_full_depth": True,
                    "interface_source": "reconstructed_from_level_centers",
                    "layer_thickness_source": "nominal interfaces",
                    "time_varying_layer_thickness_used": False,
                    "vertical_coordinate": (
                        "nominal_layer_interfaces_reconstructed_from_level"
                    ),
                    "source_vertical_level_count": 75,
                    "land_policy": "NaN over land",
                    "bottom_policy": "NaN over shallow water",
                    "missing_policy": "contiguous finite column",
                    "provenance": "MOM6 Boussinesq heat content",
                },
            )
        },
        coords={
            "time": target.source.time,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        },
    )


def _density_dataset(target: Anemoi) -> xr.Dataset:
    values = np.array(
        [
            [[np.nan, 1024.0], [1025.0, 1026.0]],
            [[np.nan, 1025.0], [1026.0, 1027.0]],
        ],
        dtype=np.float64,
    )
    return xr.Dataset(
        {
            "rho": (
                ("time", "level", "latitude", "longitude"),
                values[None, ...],
                {
                    "diagnostic": "ocean_density",
                    "units": "kg m-3",
                    "density_kind": "potential",
                    "reference_pressure_pa": 0.0,
                    "equation_of_state": "wright_full",
                    "subtract_1000": False,
                    "sigma_offset_kg_m3": 0.0,
                    "source_temperature": "temp",
                    "source_salinity": "so",
                    "vertical_coordinate": "nominal_layer_center_depth",
                    "source_vertical_level_count": 75,
                    "land_policy": "NaN over land",
                    "missing_policy": "elementwise finite inputs",
                    "provenance": "Wright (1997)",
                },
            )
        },
        coords={
            "time": target.source.time,
            "level": [5.0, 50.0],
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        },
    )


def _assert_sample_stats(
    packed: xr.Dataset,
    variable_name: str,
    expected_values: np.ndarray,
) -> None:
    variable_idx = packed.attrs["variables"].index(variable_name)
    selector = {"time": 0, "variable": variable_idx, "ensemble": 0}
    valid = np.asarray(expected_values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]

    assert packed["count_array"].isel(**selector).compute().item() == valid.size
    assert packed["has_nans_array"].isel(**selector).compute().item() == (
        valid.size != np.asarray(expected_values).size
    )
    assert packed["maximum_array"].isel(**selector).compute().item() == valid.max()
    assert packed["minimum_array"].isel(**selector).compute().item() == valid.min()
    assert packed["sums_array"].isel(**selector).compute().item() == valid.sum()
    assert packed["squares_array"].isel(**selector).compute().item() == (valid**2).sum()


def test_mld_packs_as_one_unsuffixed_float32_channel_with_aligned_stats():
    target = _target()
    source = _mld_dataset(target)

    packed = target.apply_transforms_to_sample(source)

    assert packed.attrs["variables"] == ["mld"]
    assert packed["data"].dtype == np.float32
    metadata = packed.attrs["variables_metadata"]["mld"]
    assert metadata["density_threshold"] == 0.03
    assert metadata["equation_of_state"] == "wright_full"
    assert metadata["vertical_coordinate"] == "nominal_layer_center_depth"
    assert metadata["source_vertical_level_count"] == 75
    np.testing.assert_allclose(
        packed["data"].isel(time=0, variable=0, ensemble=0),
        source["mld"].values.reshape(-1),
        equal_nan=True,
    )
    _assert_sample_stats(packed, "mld", source["mld"].values)
    for stat_name in target.packed_stat_names:
        assert packed[stat_name].sizes["variable"] == 1


def test_ohc_packs_once_without_rescaling_and_preserves_gigajoule_metadata():
    target = _target()
    source = _ohc_dataset(target)

    packed = target.apply_transforms_to_sample(source)

    assert packed.attrs["variables"] == ["ohc300"]
    assert packed["data"].dtype == np.float32
    metadata = packed.attrs["variables_metadata"]["ohc300"]
    assert metadata["diagnostic"] == "ocean_heat_content"
    assert metadata["units"] == "GJ m-2"
    assert metadata["native_units"] == "J m-2"
    assert metadata["joules_to_output_units"] == 1.0e-9
    assert metadata["integration_depth"] == 300.0
    assert metadata["require_full_depth"] is True
    assert metadata["time_varying_layer_thickness_used"] is False
    np.testing.assert_allclose(
        packed["data"].isel(time=0, variable=0, ensemble=0),
        source["ohc300"].values.reshape(-1),
        equal_nan=True,
    )


def test_density_metadata_is_copied_to_every_expanded_level_channel():
    target = _target()

    packed = target.apply_transforms_to_sample(_density_dataset(target))

    assert packed.attrs["variables"] == ["rho_0", "rho_1"]
    for name in packed.attrs["variables"]:
        metadata = packed.attrs["variables_metadata"][name]
        assert metadata["diagnostic"] == "ocean_density"
        assert metadata["units"] == "kg m-3"
        assert metadata["density_kind"] == "potential"
        assert metadata["reference_pressure_pa"] == 0.0
        assert metadata["equation_of_state"] == "wright_full"


def test_mld_merges_after_atmosphere_preserving_the_existing_source_block():
    target = _target()
    atmosphere = target.apply_transforms_to_sample(_atmosphere_dataset(target))
    ocean = target.apply_transforms_to_sample(_mld_dataset(target))
    atmosphere_names = list(atmosphere.attrs["variables"])

    merged = target.merge_multisource([atmosphere, ocean])

    assert merged.attrs["variables"] == atmosphere_names + ["mld"]
    assert merged.attrs["variables"][: len(atmosphere_names)] == atmosphere_names
    assert list(merged.attrs["variables_metadata"]) == merged.attrs["variables"]
    np.testing.assert_array_equal(
        merged["variable"],
        np.arange(len(merged.attrs["variables"])),
    )
    _assert_sample_stats(merged, "mld", _mld_dataset(target)["mld"].values)
    for stat_name in target.packed_stat_names:
        assert merged[stat_name].sizes["variable"] == len(merged.attrs["variables"])


def test_uo_surface_mask_masks_mld_and_refreshes_its_sample_stats():
    target = _target(
        multisource_masks={
            "mask_from": "uo_0",
            "mask_variables": ["mld"],
        }
    )
    atmosphere = target.apply_transforms_to_sample(_atmosphere_dataset(target))
    mld_values = np.array([[10.0, 20.0], [30.0, 40.0]])
    uo_values = np.array([[np.nan, 1.0], [1.0, 1.0]])
    ocean = target.apply_transforms_to_sample(
        _mld_dataset(target, values=mld_values, uo_values=uo_values)
    )

    _assert_sample_stats(ocean, "mld", mld_values)
    merged = target.merge_multisource([atmosphere, ocean])

    expected_mld = np.array([[np.nan, 20.0], [30.0, 40.0]])
    mld_idx = merged.attrs["variables"].index("mld")
    np.testing.assert_allclose(
        merged["data"].isel(time=0, variable=mld_idx, ensemble=0),
        expected_mld.reshape(-1),
        equal_nan=True,
    )
    _assert_sample_stats(merged, "mld", expected_mld)


def test_multisource_land_mask_is_stored_as_static_noncomputed_data():
    mask_config = {
        "mask_from": "uo_0",
        "mask_variables": ["mld"],
        "output": "ocean_land",
        "land_value": 1,
        "ocean_value": 0,
    }
    target = _target(multisource_masks=mask_config)
    atmosphere = target.apply_transforms_to_sample(_atmosphere_dataset(target))
    ocean = target.apply_transforms_to_sample(
        _mld_dataset(
            target,
            uo_values=np.array([[np.nan, 1.0], [1.0, 1.0]]),
        )
    )

    merged = target.merge_multisource([atmosphere, ocean])
    metadata = merged.attrs["variables_metadata"]["ocean_land"]
    assert metadata == {
        "computed_forcing": False,
        "constant_in_time": True,
    }
    mask_idx = merged.attrs["variables"].index("ocean_land")
    np.testing.assert_array_equal(
        merged["data"].isel(time=0, variable=mask_idx, ensemble=0),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    placeholder = target._add_multisource_mask_placeholders(
        atmosphere.copy(deep=True)
    )
    assert placeholder.attrs["variables_metadata"]["ocean_land"] == metadata


def test_expand_anemoi_dataset_restores_mld_as_a_two_dimensional_field():
    target = _target()
    atmosphere = target.apply_transforms_to_sample(_atmosphere_dataset(target))
    ocean_source = _mld_dataset(target)
    ocean = target.apply_transforms_to_sample(ocean_source)
    merged = target.merge_multisource([atmosphere, ocean])
    merged["dates"] = xr.DataArray(target.source.time, dims="time")

    expanded = expand_anemoi_dataset(
        merged,
        "data",
        merged.attrs["variables"],
    )

    assert "mld" in expanded
    assert "level" not in expanded["mld"].dims
    assert expanded["mld"].dims == ("time", "ensemble", "cell")
    np.testing.assert_allclose(
        expanded["mld"].isel(time=0, ensemble=0),
        ocean_source["mld"].values.reshape(-1),
        equal_nan=True,
    )
