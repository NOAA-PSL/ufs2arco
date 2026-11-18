import numpy as np
import pytest
import xarray as xr

from ufs2arco.layers2pressure import Layers2Pressure
from ufs2arco.transforms.layers_to_pressure import (
    _interpolate_with_replay_fallback,
    layers_to_pressure,
)
from ufs2arco.transforms.transformer import Transformer


def _replay_dataset() -> xr.Dataset:
    converter = Layers2Pressure(level_name="level", interface_name="phalf")
    level = converter.pfull.values
    coords_3d = {
        "time": [np.datetime64("2017-01-01")],
        "level": level,
        "latitude": [0.0],
        "longitude": [0.0, 90.0],
    }
    pressfc = xr.DataArray(
        [[[[100_000.0, 70_000.0]]]],
        dims=("time", "surface", "latitude", "longitude"),
        coords={
            "time": coords_3d["time"],
            "surface": [0],
            "latitude": coords_3d["latitude"],
            "longitude": coords_3d["longitude"],
        },
        attrs={"units": "Pa"},
        name="pressfc",
    ).squeeze("surface", drop=True)
    shape = (1, level.size, 1, 2)
    temperature = xr.DataArray(
        np.full(shape, 280.0),
        dims=("time", "level", "latitude", "longitude"),
        coords=coords_3d,
        attrs={"units": "K"},
        name="tmp",
    )
    humidity = xr.DataArray(
        np.full(shape, 0.001),
        dims=temperature.dims,
        coords=coords_3d,
        attrs={"units": "kg/kg"},
        name="spfh",
    )
    thickness = converter.calc_delz(pressfc, temperature, humidity).rename("delz")
    thickness.attrs["units"] = "m"
    diagnosed_pressure = converter.calc_layer_mean_pressure(
        pressfc,
        temperature,
        humidity,
        thickness,
    )
    ugrd = (4.0 + 1.5 * np.log(diagnosed_pressure)).transpose(
        *temperature.dims
    ).rename("ugrd")
    ugrd.attrs["units"] = "m s-1"
    vgrd = (-2.0 + 0.5 * np.log(diagnosed_pressure)).transpose(
        *temperature.dims
    ).rename("vgrd")
    vgrd.attrs["units"] = "m s-1"

    xds = xr.Dataset(
        {
            "pressfc": pressfc,
            "tmp": temperature,
            "spfh": humidity,
            "delz": thickness,
            "ugrd": ugrd,
            "vgrd": vgrd,
        },
        attrs={"source": "synthetic replay"},
    )
    xds["level"].attrs["units"] = "hPa"
    return xds


def _interpolate_profile(
    pressures_hpa: list[float],
    values: list[float],
    target_hpa: float = 850.0,
) -> float:
    converter = Layers2Pressure(level_name="level", interface_name="phalf")
    level = np.arange(len(pressures_hpa))
    pressure = xr.DataArray(
        np.asarray(pressures_hpa) * 100.0,
        dims="level",
        coords={"level": level},
    )
    source = xr.DataArray(
        values,
        dims="level",
        coords={"level": level},
    )
    return _interpolate_with_replay_fallback(
        converter,
        source,
        target_hpa * 100.0,
        pressure,
    ).item()


def _log_pressure_field(pressure_hpa: float) -> float:
    return 3.0 + 2.5 * np.log(pressure_hpa * 100.0)


def test_transform_interpolates_true_pressure_and_masks_below_ground():
    xds = _replay_dataset()

    result = layers_to_pressure(
        xds,
        variables={"ugrd": "ugrd_pl"},
        levels_hpa=[200, 850],
    )

    assert list(result.data_vars) == ["ugrd_pl"]
    assert result["ugrd_pl"].dims == ("time", "level", "latitude", "longitude")
    np.testing.assert_array_equal(result["level"], [200, 850])
    assert np.issubdtype(result["level"].dtype, np.integer)
    expected_200 = 4.0 + 1.5 * np.log(20_000.0)
    expected_850 = 4.0 + 1.5 * np.log(85_000.0)
    assert result["ugrd_pl"].sel(level=200, longitude=0.0).item() == pytest.approx(
        expected_200,
        rel=1e-5,
    )
    assert result["ugrd_pl"].sel(level=850, longitude=0.0).item() == pytest.approx(
        expected_850,
        rel=1e-5,
    )
    assert np.isnan(result["ugrd_pl"].sel(level=850, longitude=90.0).item())
    assert result["level"].attrs["units"] == "hPa"
    assert result["level"].attrs["type_of_level"] == "isobaricInhPa"
    assert result["ugrd_pl"].attrs["units"] == "m s-1"
    assert result["ugrd_pl"].attrs["source_variable"] == "ugrd"
    assert result["ugrd_pl"].attrs["vertical_interpolation"] == "linear_in_log_pressure"
    assert result["ugrd_pl"].attrs["source_vertical_grid"] == "ufs_replay_127_layer_hybrid"
    assert result["ugrd_pl"].attrs["support_variables"] == [
        "pressfc",
        "tmp",
        "spfh",
        "delz",
    ]
    assert result["ugrd_pl"].attrs["below_ground_policy"] == "mask"
    assert result["ugrd_pl"].attrs["unbracketed_policy"] == "bounded_recovery"
    assert (
        result["ugrd_pl"].attrs["extrapolation_policy"]
        == "linear_in_log_pressure_from_lower_pressure_pair_within_10_hpa"
    )
    assert (
        result["ugrd_pl"].attrs["nearest_neighbor_policy"]
        == "single_lower_pressure_level_within_10_hpa"
    )
    assert result["ugrd_pl"].attrs["max_fallback_distance_hpa"] == 10.0
    assert result["ugrd_pl"].attrs["max_bridge_span_hpa"] == 25.0
    assert result["ugrd_pl"].attrs["max_bridged_missing_levels"] == 1
    assert result.attrs["source"] == "synthetic replay"


def test_coefficients_are_source_specific(monkeypatch):
    xds = _replay_dataset()
    calls = []
    original = Layers2Pressure.get_interp_coefficients

    def record_call(self, pstar, prsl, include_masks=True):
        calls.append(pstar)
        return original(self, pstar, prsl, include_masks=include_masks)

    monkeypatch.setattr(Layers2Pressure, "get_interp_coefficients", record_call)

    result = layers_to_pressure(
        xds,
        variables={"ugrd": "ugrd_pl", "vgrd": "vgrd_pl"},
        levels_hpa=[200, 250, 850],
    )

    assert list(result.data_vars) == ["ugrd_pl", "vgrd_pl"]
    assert calls == [
        20_000.0,
        20_000.0,
        25_000.0,
        25_000.0,
        85_000.0,
        85_000.0,
    ]


def test_normal_bracket_takes_precedence_over_fallbacks():
    actual = _interpolate_profile(
        [847.8, 857.5, 866.9],
        [100.0, 30.0, 1_000.0],
    )
    expected_factor = (
        (np.log(850.0) - np.log(847.8))
        / (np.log(857.5) - np.log(847.8))
    )
    expected = 100.0 + expected_factor * (30.0 - 100.0)

    assert actual == pytest.approx(expected)


def test_bridge_skips_one_unavailable_native_level():
    actual = _interpolate_profile(
        [847.8, 857.5, 866.9],
        [_log_pressure_field(847.8), np.nan, _log_pressure_field(866.9)],
    )

    assert actual == pytest.approx(_log_pressure_field(850.0))


def test_extrapolates_from_adjacent_levels_within_10_hpa():
    actual = _interpolate_profile(
        [837.6, 847.8, 857.5],
        [_log_pressure_field(837.6), _log_pressure_field(847.8), np.nan],
    )

    assert actual == pytest.approx(_log_pressure_field(850.0))


@pytest.mark.parametrize(
    ("pressures_hpa", "expected_finite"),
    [([830.0, 840.0, 870.0], True), ([829.99, 839.99, 870.0], False)],
)
def test_extrapolation_uses_inclusive_10_hpa_limit(
    pressures_hpa,
    expected_finite,
):
    actual = _interpolate_profile(
        pressures_hpa,
        [
            _log_pressure_field(pressures_hpa[0]),
            _log_pressure_field(pressures_hpa[1]),
            np.nan,
        ],
    )

    assert bool(np.isfinite(actual)) is expected_finite
    if expected_finite:
        assert actual == pytest.approx(_log_pressure_field(850.0))


@pytest.mark.parametrize(
    ("available_pressure_hpa", "expected_finite"),
    [(847.8, True), (840.0, True), (839.99, False)],
)
def test_nearest_fallback_uses_inclusive_10_hpa_limit(
    available_pressure_hpa,
    expected_finite,
):
    actual = _interpolate_profile(
        [830.0, available_pressure_hpa, 870.0],
        [np.nan, 42.0, np.nan],
    )

    assert bool(np.isfinite(actual)) is expected_finite
    if expected_finite:
        assert actual == 42.0


def test_upper_pressure_value_alone_does_not_trigger_fallback():
    actual = _interpolate_profile(
        [830.0, 840.0, 860.0],
        [np.nan, np.nan, 42.0],
    )

    assert np.isnan(actual)


def test_split_level_chunks_match_eager_fallbacks():
    converter = Layers2Pressure(level_name="level", interface_name="phalf")
    pressures_hpa = np.array([837.6, 847.8, 857.5, 866.9])
    pressure = xr.DataArray(
        np.broadcast_to(pressures_hpa, (4, 4)) * 100.0,
        dims=("case", "level"),
        coords={"case": np.arange(4), "level": np.arange(4)},
    )
    log_values = np.asarray([_log_pressure_field(value) for value in pressures_hpa])
    source = xr.DataArray(
        np.asarray(
            [
                log_values,
                [log_values[0], log_values[1], np.nan, log_values[3]],
                [log_values[0], log_values[1], np.nan, np.nan],
                [np.nan, 42.0, np.nan, np.nan],
            ]
        ),
        dims=pressure.dims,
        coords=pressure.coords,
    )
    expected = _interpolate_with_replay_fallback(
        converter,
        source,
        85_000.0,
        pressure,
    )

    actual = _interpolate_with_replay_fallback(
        converter,
        source.chunk({"case": 2, "level": 1}),
        85_000.0,
        pressure.chunk({"case": 2, "level": 1}),
    ).compute()

    xr.testing.assert_allclose(actual, expected)


def test_selected_pressure_inversion_disables_fallbacks():
    actual = _interpolate_profile(
        [860.0, 840.0],
        [1.0, 2.0],
    )

    assert np.isnan(actual)


def test_nearest_fallback_rejects_an_adjacent_pressure_inversion():
    actual = _interpolate_profile(
        [860.0, 840.0],
        [1.0, 2.0],
        target_hpa=870.0,
    )

    assert np.isnan(actual)


def test_surface_pressure_guard_applies_after_fallback(monkeypatch):
    xds = _replay_dataset()
    source = xds["ugrd"].copy(deep=True)
    source.values[:] = np.nan
    pressure = xr.full_like(source, np.nan)
    indices = [100, 101, 102]
    for index, pressure_hpa in zip(indices, [837.6, 847.8, 857.5]):
        pressure.values[:, index, :, :] = pressure_hpa * 100.0
        if pressure_hpa < 850.0:
            source.values[:, index, :, :] = _log_pressure_field(pressure_hpa)
    xds["ugrd"] = source
    monkeypatch.setattr(
        Layers2Pressure,
        "calc_layer_mean_pressure",
        lambda self, pressfc, temp, spfh, delz: pressure,
    )

    result = layers_to_pressure(
        xds,
        variables={"ugrd": "ugrd_pl"},
        levels_hpa=[850],
    )

    assert result["ugrd_pl"].sel(longitude=0.0).item() == pytest.approx(
        _log_pressure_field(850.0)
    )
    assert np.isnan(result["ugrd_pl"].sel(longitude=90.0).item())


@pytest.mark.parametrize("coordinate_problem", ["subset", "reversed", "duplicate"])
def test_transform_rejects_incomplete_or_misordered_replay_grid(coordinate_problem):
    xds = _replay_dataset()
    if coordinate_problem == "subset":
        xds = xds.isel(level=slice(1, None))
    elif coordinate_problem == "reversed":
        xds = xds.isel(level=slice(None, None, -1))
    else:
        level = xds["level"].values.copy()
        level[1] = level[0]
        xds = xds.assign_coords(level=level)

    with pytest.raises(ValueError, match="127-level|does not match"):
        layers_to_pressure(
            xds,
            variables={"ugrd": "ugrd_pl"},
            levels_hpa=[850],
        )


@pytest.mark.parametrize("levels", [[850, 850], [850.5], [85_000], []])
def test_transform_rejects_invalid_target_levels(levels):
    with pytest.raises(ValueError, match="levels_hpa"):
        layers_to_pressure(
            _replay_dataset(),
            variables={"ugrd": "ugrd_pl"},
            levels_hpa=levels,
        )


def test_transform_rejects_missing_support_and_output_collisions():
    xds = _replay_dataset()
    with pytest.raises(KeyError, match="missing support"):
        layers_to_pressure(
            xds.drop_vars("spfh"),
            variables={"ugrd": "ugrd_pl"},
            levels_hpa=[850],
        )

    xds["ugrd_pl"] = xds["ugrd"]
    with pytest.raises(ValueError, match="already exist"):
        layers_to_pressure(
            xds,
            variables={"ugrd": "ugrd_pl"},
            levels_hpa=[850],
        )


def test_transform_rejects_nonnegative_native_layer_thickness():
    xds = _replay_dataset()
    xds["delz"] = xds["delz"].copy()
    xds["delz"].values[0, 10, 0, 0] = 0.0

    with pytest.raises(ValueError, match="negative layer thicknesses"):
        layers_to_pressure(
            xds,
            variables={"ugrd": "ugrd_pl"},
            levels_hpa=[850],
        )


def test_transform_rejects_integer_surface_pressure():
    xds = _replay_dataset()
    xds["pressfc"] = xds["pressfc"].astype(np.int64)
    xds["pressfc"].attrs["units"] = "Pa"

    with pytest.raises(TypeError, match="surface pressure.*floating point"):
        layers_to_pressure(
            xds,
            variables={"ugrd": "ugrd_pl"},
            levels_hpa=[850],
        )


def test_unrelated_bottom_pressure_inversion_does_not_mask_upper_level():
    xds = _replay_dataset()
    bottom_level = xds["level"].values[-1]
    xds["delz"].loc[{"level": bottom_level}] *= 10.0

    result = layers_to_pressure(
        xds,
        variables={"ugrd": "ugrd_pl"},
        levels_hpa=[200],
    )

    assert np.isfinite(result["ugrd_pl"].sel(level=200)).all()


def test_selected_pressure_bracket_inversion_is_masked(monkeypatch):
    xds = _replay_dataset()
    converter = Layers2Pressure(level_name="level", interface_name="phalf")
    diagnosed_pressure = converter.calc_layer_mean_pressure(
        xds["pressfc"],
        xds["tmp"],
        xds["spfh"],
        xds["delz"],
    ).transpose(*xds["tmp"].dims)
    coefficients = converter.get_interp_coefficients(
        20_000.0,
        diagnosed_pressure,
        include_masks=False,
    )
    left_index = int(coefficients["left_index"].isel(time=0, latitude=0, longitude=0))
    right_index = int(
        coefficients["right_index"].isel(time=0, latitude=0, longitude=0)
    )
    assert right_index == left_index + 1

    inverted_pressure = diagnosed_pressure.copy(deep=True)
    inverted_pressure.values[:, [left_index, right_index], :, :] = (
        inverted_pressure.values[:, [right_index, left_index], :, :]
    )
    monkeypatch.setattr(
        Layers2Pressure,
        "calc_layer_mean_pressure",
        lambda self, pressfc, temp, spfh, delz: inverted_pressure,
    )

    result = layers_to_pressure(
        xds,
        variables={"ugrd": "ugrd_pl"},
        levels_hpa=[200],
    )

    assert result["ugrd_pl"].sel(level=200).isnull().all()


def test_transformer_rejects_pressure_and_finite_volume_in_same_branch():
    with pytest.raises(ValueError, match="separate source branch"):
        Transformer(
            {
                "layers_to_pressure": {
                    "variables": {"ugrd": "ugrd_pl"},
                    "levels_hpa": [850],
                },
                "fv_vertical_regrid": {
                    "weight_var": "delz",
                    "interfaces": [0, 1000],
                },
            }
        )


def test_transformer_runs_pressure_transform_after_vector_rotation(monkeypatch):
    calls = []

    def rotate(xds, **kwargs):
        calls.append("rotate")
        return xds

    def pressure(xds, **kwargs):
        calls.append("pressure")
        return xds

    monkeypatch.setattr("ufs2arco.transforms.transformer.rotate_vectors", rotate)
    monkeypatch.setattr("ufs2arco.transforms.transformer.layers_to_pressure", pressure)
    transformer = Transformer(
        {
            "layers_to_pressure": {},
            "rotate_vectors": {},
        }
    )

    transformer(xr.Dataset())

    assert calls == ["rotate", "pressure"]
