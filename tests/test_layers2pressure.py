import numpy as np
import pytest
import xarray as xr

from ufs2arco.layers2pressure import Layers2Pressure


def _converter() -> Layers2Pressure:
    return Layers2Pressure(
        ak=np.array([10_000.0, 50_000.0, 100_000.0]),
        bk=np.zeros(3),
        pfull=np.array([300.0, 750.0]),
        phalf=np.array([100.0, 500.0, 1000.0]),
        level_name="level",
        interface_name="phalf",
    )


def test_indexed_interpolation_uses_two_log_pressure_brackets():
    converter = _converter()
    pressure = xr.DataArray(
        np.array(
            [
                [10_000.0, 10_000.0, 60_000.0, 10_000.0],
                [30_000.0, 50_000.0, 70_000.0, 30_000.0],
                [60_000.0, 70_000.0, 80_000.0, 60_000.0],
                [90_000.0, 90_000.0, 90_000.0, 90_000.0],
            ]
        ),
        dims=("level", "column"),
        coords={"level": [1, 2, 3, 4], "column": [0, 1, 2, 3]},
    )
    field = 3.0 + 2.5 * np.log(pressure)
    field.loc[{"level": 3, "column": 3}] = np.nan

    coefficients = converter.get_interp_coefficients(
        50_000.0,
        pressure,
        include_masks=False,
    )
    result = converter.interp2pressure(
        field,
        50_000.0,
        pressure,
        cds=coefficients,
    )

    expected = 3.0 + 2.5 * np.log(50_000.0)
    assert result.sel(column=0).item() == pytest.approx(expected)
    assert result.sel(column=1).item() == pytest.approx(expected)
    assert np.isnan(result.sel(column=2).item())
    assert np.isnan(result.sel(column=3).item())
    assert coefficients["valid"].sel(column=0).item()
    assert coefficients["exact"].sel(column=1).item()
    assert not coefficients["valid"].sel(column=2).item()
    assert "level" not in coefficients["left_index"].dims
    assert "level" not in coefficients["right_index"].dims
    assert "level" not in result.coords
    assert 0.0 <= coefficients["factor"].sel(column=0).item() <= 1.0


def test_default_coefficients_preserve_legacy_masks():
    converter = _converter()
    pressure = xr.DataArray(
        [10_000.0, 30_000.0, 60_000.0],
        dims="level",
        coords={"level": [1, 2, 3]},
    )

    coefficients = converter.get_interp_coefficients(50_000.0, pressure)

    assert "is_left" in coefficients
    assert "is_right" in coefficients
    assert coefficients["is_left"].sum().item() == 1
    assert coefficients["is_right"].sum().item() == 1


def test_missing_intermediate_pressure_does_not_create_a_nonadjacent_bracket():
    converter = _converter()
    pressure = xr.DataArray(
        [10_000.0, np.nan, 60_000.0, 90_000.0],
        dims="level",
        coords={"level": [1, 2, 3, 4]},
    )
    field = 3.0 + 2.5 * np.log(pressure)

    coefficients = converter.get_interp_coefficients(
        50_000.0,
        pressure,
        include_masks=False,
    )
    result = converter.interp2pressure(
        field,
        50_000.0,
        pressure,
        cds=coefficients,
    )

    assert not coefficients["valid"].item()
    assert np.isnan(result.item())


def test_indexed_interpolation_supports_descending_pressure_order():
    converter = _converter()
    pressure = xr.DataArray(
        [90_000.0, 60_000.0, 30_000.0, 10_000.0],
        dims="level",
        coords={"level": [4, 3, 2, 1]},
    )
    field = 3.0 + 2.5 * np.log(pressure)

    result = converter.interp2pressure(field, 50_000.0, pressure)

    expected = 3.0 + 2.5 * np.log(50_000.0)
    assert result.item() == pytest.approx(expected)


@pytest.mark.parametrize("pstar", [0.0, -1.0, np.nan, np.inf])
def test_coefficients_reject_invalid_target_pressure(pstar):
    converter = _converter()
    pressure = xr.DataArray([10_000.0, 50_000.0], dims="level")

    with pytest.raises(ValueError, match="positive and finite"):
        converter.get_interp_coefficients(pstar, pressure)


def test_missing_humidity_remains_missing_in_diagnosed_thickness_and_pressure():
    converter = _converter()
    pressfc = xr.DataArray([100_000.0], dims="column", attrs={"units": "Pa"})
    temperature = xr.DataArray(
        [[280.0], [270.0]],
        dims=("level", "column"),
        coords={"level": [300.0, 750.0]},
    )
    humidity = xr.DataArray(
        [[0.001], [np.nan]],
        dims=("level", "column"),
        coords={"level": [300.0, 750.0]},
    )

    thickness = converter.calc_delz(pressfc, temperature, humidity)
    pressure = converter.calc_layer_mean_pressure(
        pressfc,
        temperature,
        humidity,
        thickness,
    )

    assert np.isfinite(thickness.sel(level=300.0).item())
    assert np.isnan(thickness.sel(level=750.0).item())
    assert np.isnan(pressure.sel(level=750.0).item())
