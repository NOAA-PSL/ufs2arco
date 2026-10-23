import numpy as np
import pytest

from ufs2arco.ocean_diagnostics import (
    GRAVITY,
    HEAT_CAPACITY,
    RHO_0,
    _ocean_heat_content_for_depths,
    _validated_ohc_depths,
    hydrostatic_pressure,
    interfaces_from_centers,
    mixed_layer_depth_by_density_difference,
    ocean_heat_content,
    wright_density,
)


def _vectorized_reference(theta, salt, depth, threshold):
    """Reproduce the linked branch's full-column NumPy algorithm."""
    theta = np.asarray(theta, dtype=np.float64)
    salt = np.asarray(salt, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)

    finite = np.isfinite(theta) & np.isfinite(salt)
    valid = np.logical_and.accumulate(finite, axis=-1)
    density = wright_density(theta, salt, 0.0)
    delta = density - density[..., :1]
    delta = np.where(valid, delta, -1.0e30)

    rise = np.diff(delta, axis=-1)
    crossing = (
        (rise > 0.0)
        & (delta[..., :-1] < threshold)
        & (delta[..., 1:] >= threshold)
    )
    found = crossing.any(axis=-1)
    index = np.argmax(crossing, axis=-1)
    previous = np.take_along_axis(
        delta[..., :-1],
        index[..., None],
        axis=-1,
    )[..., 0]
    current = np.take_along_axis(
        delta[..., 1:],
        index[..., None],
        axis=-1,
    )[..., 0]
    denominator = np.where(found, current - previous, 1.0)
    fraction = (threshold - previous) / denominator
    interpolated = (
        fraction * depth[index + 1]
        + (1.0 - fraction) * depth[index]
    )

    valid_count = valid.sum(axis=-1)
    is_ocean = valid_count > 0
    deepest_index = np.clip(valid_count - 1, 0, depth.size - 1)
    deepest_delta = np.take_along_axis(
        delta,
        deepest_index[..., None],
        axis=-1,
    )[..., 0]
    reaches_bottom = (~found) & is_ocean & (deepest_delta < threshold)

    result = np.zeros(found.shape, dtype=np.float64)
    result = np.where(found, interpolated, result)
    result = np.where(reaches_bottom, depth[deepest_index], result)
    pathological = (~found) & (~reaches_bottom) & is_ocean
    result = np.where(pathological, np.nan, result)
    return np.where(is_ocean, result, np.nan)


def _vectorized_ohc_reference(
    theta,
    interfaces,
    max_depth,
    require_full_depth,
):
    """Reproduce the linked branch's full-column OHC algorithm."""
    theta = np.asarray(theta, dtype=np.float64)
    interfaces = np.asarray(interfaces, dtype=np.float64)
    valid = np.logical_and.accumulate(np.isfinite(theta), axis=-1)
    thickness = np.diff(interfaces)
    if max_depth is None:
        included = thickness
    else:
        included = np.clip(max_depth - interfaces[:-1], 0.0, thickness)
    integral = np.where(
        valid,
        np.nan_to_num(theta, nan=0.0) * included,
        0.0,
    ).sum(axis=-1)
    valid_count = valid.sum(axis=-1)
    keep = valid_count > 0
    if max_depth is not None and require_full_depth:
        keep &= interfaces[valid_count] >= max_depth
    return np.where(keep, RHO_0 * HEAT_CAPACITY * integral, np.nan)


@pytest.mark.parametrize(
    "temperature,salinity,expected",
    [
        (5.0, 0.0, 999.9694329140925),
        (5.0, 35.0, 1027.649216275557),
        (25.0, 35.0, 1023.3480875706816),
        (20.0, 35.0, 1024.7634359926774),
        (10.0, 35.0, 1026.9374762527486),
    ],
)
def test_wright_full_reference_values(temperature, salinity, expected):
    assert wright_density(temperature, salinity, 0.0) == pytest.approx(
        expected,
        abs=1.0e-10,
    )


@pytest.mark.parametrize(
    "eos,temperature_bounds,salinity_bounds,maximum_pressure",
    [
        ("wright_full", [-2.0, 40.0], [0.0, 40.0], 1.0e8),
        ("wright_red", [-2.0, 30.0], [28.0, 38.0], 5.0e7),
    ],
)
def test_wright_range_check_uses_selected_eos_bounds(
    eos,
    temperature_bounds,
    salinity_bounds,
    maximum_pressure,
    caplog,
):
    temperature = np.array(temperature_bounds)
    salinity = np.array(salinity_bounds)
    pressure = np.array([0.0, maximum_pressure])
    wright_density(temperature, salinity, pressure, eos=eos, check_range=True)
    assert not caplog.records

    temperature += [-1.0, 1.0]
    salinity += [-1.0, 1.0]
    pressure += [-1.0, 1.0]
    result = wright_density(
        temperature,
        salinity,
        pressure,
        eos=eos,
        check_range=True,
    )

    np.testing.assert_array_equal(
        result,
        wright_density(temperature, salinity, pressure, eos=eos),
    )
    assert len(caplog.records) == 3
    for record, name in zip(
        caplog.records,
        ("potential temperature", "salinity", "pressure"),
    ):
        assert f"wright_density: {name} range" in record.message
        assert f"the {eos} fit range" in record.message
        assert "densities there are extrapolated" in record.message


def test_wright_range_check_accepts_fifth_positional_argument(caplog):
    result = wright_density(45.0, 35.0, 0.0, "wright_full", True)

    assert result == pytest.approx(1015.8680111248782)
    assert "potential temperature range" in caplog.text


@pytest.mark.parametrize("kwargs", [{}, {"check_range": False}])
def test_wright_disabled_range_check_preserves_laziness_and_silence(kwargs, caplog):
    da = pytest.importorskip("dask.array")
    from dask import delayed

    calls = []

    @delayed
    def read_temperature():
        calls.append(True)
        return np.array([45.0], dtype=np.float32)

    temperature = da.from_delayed(
        read_temperature(),
        shape=(1,),
        dtype=np.float32,
    )
    result = wright_density(temperature, 35.0, 0.0, **kwargs)

    assert not calls
    assert result.dtype == np.float64
    assert result.chunks == temperature.chunks
    np.testing.assert_allclose(result.compute(), [1015.8680111248782])
    assert calls == [True]
    assert not caplog.records


@pytest.mark.parametrize(
    "threshold,expected",
    [
        (0.03, 15.137991924763236),
        (0.125, 15.574966353180153),
    ],
)
def test_mld_matches_frozen_threshold_interpolation(threshold, expected):
    depth = np.array([5.0, 15.0, 25.0, 35.0])
    temperature = np.array([20.0, 20.0, 10.0, 10.0])
    salinity = np.full(depth.size, 35.0)

    result = mixed_layer_depth_by_density_difference(
        temperature,
        salinity,
        depth,
        threshold,
    )

    assert result == pytest.approx(expected, abs=1.0e-10)


def test_exact_threshold_crossing_selects_the_deeper_center():
    depth = np.array([5.0, 15.0, 25.0, 35.0])
    temperature = np.array([20.0, 20.0, 10.0, 10.0])
    salinity = np.full(depth.size, 35.0)
    threshold = wright_density(10.0, 35.0, 0.0) - wright_density(
        20.0,
        35.0,
        0.0,
    )

    result = mixed_layer_depth_by_density_difference(
        temperature,
        salinity,
        depth,
        threshold,
    )

    assert result == pytest.approx(25.0)


@pytest.mark.parametrize(
    "temperature,salinity,expected",
    [
        ([20.0, 20.0, 20.0, 20.0], [35.0] * 4, 35.0),
        ([20.0, 20.0, np.nan, np.nan], [35.0, 35.0, np.nan, np.nan], 15.0),
        ([20.0, 20.0, np.nan, 10.0], [35.0, 35.0, np.nan, 35.0], 15.0),
    ],
)
def test_bottom_fallback_uses_the_deepest_contiguous_valid_center(
    temperature,
    salinity,
    expected,
):
    result = mixed_layer_depth_by_density_difference(
        temperature,
        salinity,
        [5.0, 15.0, 25.0, 35.0],
        0.03,
    )

    assert result == pytest.approx(expected)


def test_missing_surface_is_land_even_when_deeper_values_are_finite():
    result = mixed_layer_depth_by_density_difference(
        [np.nan, 20.0, 10.0, 10.0],
        [np.nan, 35.0, 35.0, 35.0],
        [5.0, 15.0, 25.0, 35.0],
        0.03,
    )

    assert np.isnan(result)


def test_density_inversion_before_crossing_uses_the_later_rising_pair():
    result = mixed_layer_depth_by_density_difference(
        [20.0, 25.0, 25.0, 10.0],
        [35.0, 35.0, 35.0, 35.0],
        [5.0, 15.0, 25.0, 35.0],
        0.03,
    )

    assert result == pytest.approx(29.026725857851176, abs=1.0e-10)


def test_streaming_kernel_matches_linked_vectorized_algorithm():
    rng = np.random.default_rng(7)
    depth = np.array([1.0, 3.0, 7.0, 15.0, 30.0, 55.0, 90.0, 140.0])
    temperature = (
        24.0
        - np.linspace(0.0, 12.0, depth.size)[None, None, :]
        + rng.normal(0.0, 0.3, size=(2, 3, depth.size))
    ).astype(np.float32)
    salinity = (
        34.0
        + np.linspace(0.0, 1.5, depth.size)[None, None, :]
        + rng.normal(0.0, 0.02, size=(2, 3, depth.size))
    ).astype(np.float32)
    temperature[0, 1, 5:] = np.nan
    salinity[0, 1, 5:] = np.nan
    temperature[1, 0, :] = np.nan
    salinity[1, 0, :] = np.nan
    temperature[1, 2, 4] = np.nan
    salinity[1, 2, 4] = np.nan

    original_temperature = temperature.copy()
    original_salinity = salinity.copy()
    for threshold in (0.03, 0.125):
        expected = _vectorized_reference(
            temperature,
            salinity,
            depth,
            threshold,
        )
        actual = mixed_layer_depth_by_density_difference(
            temperature,
            salinity,
            depth,
            threshold,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12, equal_nan=True)

    np.testing.assert_array_equal(temperature, original_temperature)
    np.testing.assert_array_equal(salinity, original_salinity)


@pytest.mark.parametrize("threshold", [0.0, -0.03, np.nan, np.inf])
def test_invalid_thresholds_are_rejected(threshold):
    with pytest.raises(ValueError, match="finite and positive"):
        mixed_layer_depth_by_density_difference(
            [20.0, 10.0],
            [35.0, 35.0],
            [5.0, 15.0],
            threshold,
        )


@pytest.mark.parametrize(
    "depth",
    [
        [5.0],
        [5.0, np.nan],
        [5.0, 5.0],
        [15.0, 5.0],
        [-5.0, 5.0],
    ],
)
def test_invalid_depth_coordinates_are_rejected(depth):
    with pytest.raises(ValueError, match="depth|vertical levels"):
        mixed_layer_depth_by_density_difference(
            np.full(len(depth), 20.0),
            np.full(len(depth), 35.0),
            depth,
            0.03,
        )


def test_invalid_shapes_eos_and_reference_pressure_are_rejected():
    with pytest.raises(ValueError, match="vertical levels"):
        mixed_layer_depth_by_density_difference(
            [20.0, 10.0],
            [35.0, 35.0, 35.0],
            [5.0, 15.0],
            0.03,
        )
    with pytest.raises(NotImplementedError, match="equation of state"):
        mixed_layer_depth_by_density_difference(
            [20.0, 10.0],
            [35.0, 35.0],
            [5.0, 15.0],
            0.03,
            eos="teos10",
        )
    with pytest.raises(NotImplementedError, match="ref_pressure=0"):
        mixed_layer_depth_by_density_difference(
            [20.0, 10.0],
            [35.0, 35.0],
            [5.0, 15.0],
            0.03,
            ref_pressure=100_000.0,
        )


def test_hydrostatic_pressure_uses_validated_boussinesq_constants():
    depth = np.array([0.0, 10.0, 1_000.0], dtype=np.float32)

    pressure = hydrostatic_pressure(depth)

    assert pressure.dtype == np.float64
    np.testing.assert_allclose(pressure, RHO_0 * GRAVITY * depth)
    assert hydrostatic_pressure(2.0, rho0=1_000.0, gravity=10.0) == 20_000.0


@pytest.mark.parametrize("depth", [[0.0, np.nan], [-1.0], [1.0, np.inf]])
def test_hydrostatic_pressure_rejects_invalid_depth(depth):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        hydrostatic_pressure(depth)


@pytest.mark.parametrize(
    "keyword,value,error",
    [
        ("rho0", 0.0, ValueError),
        ("rho0", np.inf, ValueError),
        ("rho0", "1035", TypeError),
        ("gravity", -9.8, ValueError),
        ("gravity", True, TypeError),
    ],
)
def test_hydrostatic_pressure_rejects_invalid_constants(keyword, value, error):
    with pytest.raises(error, match=keyword):
        hydrostatic_pressure(10.0, **{keyword: value})


def test_interfaces_from_centers_reconstructs_uniform_and_variable_grids():
    np.testing.assert_array_equal(
        interfaces_from_centers([5.0, 15.0, 25.0]),
        [0.0, 10.0, 20.0, 30.0],
    )
    np.testing.assert_array_equal(
        interfaces_from_centers([1.0, 3.0, 6.0]),
        [0.0, 2.0, 4.0, 8.0],
    )


@pytest.mark.parametrize(
    "centers",
    [
        [],
        [[1.0, 2.0]],
        [1.0, np.nan],
        [-1.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [100.0, 110.0],
    ],
)
def test_interfaces_from_centers_rejects_invalid_geometry(centers):
    with pytest.raises(ValueError, match="center|interfaces"):
        interfaces_from_centers(centers)


def test_ohc300_for_uniform_temperature_is_exact():
    interfaces = np.array([0.0, 100.0, 200.0, 400.0])
    temperature = np.full(3, 10.0)

    result = ocean_heat_content(temperature, interfaces, max_depth=300.0)

    assert result == pytest.approx(RHO_0 * HEAT_CAPACITY * 10.0 * 300.0)


def test_ocean_heat_content_applies_partial_cell_weighting():
    interfaces = np.array([0.0, 100.0, 200.0, 400.0])
    temperature = np.array([1.0, 2.0, 4.0])

    result = ocean_heat_content(temperature, interfaces, max_depth=300.0)

    expected_temperature_integral = 1.0 * 100.0 + 2.0 * 100.0 + 4.0 * 100.0
    assert result == pytest.approx(
        RHO_0 * HEAT_CAPACITY * expected_temperature_integral
    )


def test_subzero_water_can_reduce_heat_content_at_greater_depth():
    interfaces = np.array([0.0, 100.0, 200.0])
    temperature = np.array([1.0, -2.0])

    shallow = ocean_heat_content(temperature, interfaces, max_depth=100.0)
    deep = ocean_heat_content(temperature, interfaces, max_depth=200.0)

    assert deep < shallow


def test_multi_depth_kernel_matches_scalar_calls_and_full_column():
    interfaces = np.array([0.0, 100.0, 200.0, 400.0])
    temperature = np.array(
        [
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 5.0],
        ],
        dtype=np.float32,
    )

    combined = _ocean_heat_content_for_depths(
        temperature,
        interfaces,
        max_depths=[50.0, 300.0, None],
    )

    assert combined.shape == (2, 3)
    assert combined.dtype == np.float64
    for index, depth in enumerate((50.0, 300.0, None)):
        expected = ocean_heat_content(temperature, interfaces, max_depth=depth)
        np.testing.assert_allclose(combined[..., index], expected)
    np.testing.assert_allclose(
        combined[..., 2],
        RHO_0
        * HEAT_CAPACITY
        * np.sum(temperature * np.diff(interfaces), axis=-1),
    )


@pytest.mark.parametrize("require_full_depth", [True, False])
def test_streaming_ohc_matches_linked_vectorized_algorithm(require_full_depth):
    rng = np.random.default_rng(11)
    interfaces = np.array([0.0, 5.0, 15.0, 40.0, 80.0, 150.0, 250.0, 400.0])
    temperature = rng.normal(8.0, 4.0, size=(2, 3, 7)).astype(np.float32)
    temperature[0, 0, :] = np.nan
    temperature[0, 1, 4:] = np.nan
    temperature[1, 2, 3] = np.nan
    original = temperature.copy()
    depths = (30.0, 300.0, None)

    actual = _ocean_heat_content_for_depths(
        temperature,
        interfaces,
        max_depths=depths,
        require_full_depth=require_full_depth,
    )

    for index, depth in enumerate(depths):
        expected = _vectorized_ohc_reference(
            temperature,
            interfaces,
            depth,
            require_full_depth,
        )
        np.testing.assert_allclose(
            actual[..., index],
            expected,
            rtol=0.0,
            atol=1.0e-6,
            equal_nan=True,
        )
    np.testing.assert_array_equal(temperature, original)


def test_multi_depth_kernel_masks_each_unreached_depth_independently():
    interfaces = np.array([0.0, 100.0, 200.0, 400.0])
    temperature = np.array([10.0, 10.0, np.nan])

    result = _ocean_heat_content_for_depths(
        temperature,
        interfaces,
        max_depths=[50.0, 300.0, None],
    )

    assert np.isfinite(result[0])
    assert np.isnan(result[1])
    assert result[2] == pytest.approx(RHO_0 * HEAT_CAPACITY * 10.0 * 200.0)


def test_ocean_heat_content_handles_land_and_shallow_columns():
    interfaces = np.array([0.0, 100.0, 200.0, 400.0])
    land = np.full(3, np.nan)
    shallow = np.array([10.0, 10.0, np.nan])

    assert np.isnan(ocean_heat_content(land, interfaces, max_depth=300.0))
    assert np.isnan(ocean_heat_content(land, interfaces, max_depth=None))
    assert np.isnan(ocean_heat_content(shallow, interfaces, max_depth=300.0))
    assert ocean_heat_content(
        shallow,
        interfaces,
        max_depth=300.0,
        require_full_depth=False,
    ) == pytest.approx(RHO_0 * HEAT_CAPACITY * 10.0 * 200.0)


def test_internal_nan_terminates_the_contiguous_water_column():
    interfaces = np.array([0.0, 100.0, 200.0, 400.0])
    temperature = np.array([10.0, np.nan, 1_000.0])

    finite_depth = ocean_heat_content(
        temperature,
        interfaces,
        max_depth=400.0,
        require_full_depth=False,
    )
    full_column = ocean_heat_content(temperature, interfaces, max_depth=None)

    expected = RHO_0 * HEAT_CAPACITY * 10.0 * 100.0
    assert finite_depth == pytest.approx(expected)
    assert full_column == pytest.approx(expected)


def test_ocean_heat_content_does_not_mutate_input_arrays():
    interfaces = np.array([0.0, 100.0, 200.0, 400.0])
    temperature = np.array([[10.0, np.nan, 2.0]], dtype=np.float32)
    original_interfaces = interfaces.copy()
    original_temperature = temperature.copy()

    _ocean_heat_content_for_depths(
        temperature,
        interfaces,
        max_depths=[50.0, 300.0, None],
        require_full_depth=False,
    )

    np.testing.assert_array_equal(interfaces, original_interfaces)
    np.testing.assert_array_equal(temperature, original_temperature)


@pytest.mark.parametrize(
    "interfaces",
    [
        [[0.0, 1.0], [2.0, 3.0]],
        [0.0, 100.0, 200.0],
        [1.0, 100.0, 200.0, 400.0],
        [0.0, 100.0, np.nan, 400.0],
        [0.0, 100.0, 100.0, 400.0],
        [0.0, 100.0, 50.0, 400.0],
    ],
)
def test_ocean_heat_content_rejects_invalid_interfaces(interfaces):
    with pytest.raises(ValueError, match="interfaces|expected"):
        ocean_heat_content([1.0, 2.0, 3.0], interfaces, max_depth=100.0)


@pytest.mark.parametrize(
    "depths,error",
    [
        ([], ValueError),
        ([300.0, 300], ValueError),
        ([None, None], ValueError),
        ([0.0], ValueError),
        ([-1.0], ValueError),
        ([np.nan], ValueError),
        ([np.inf], ValueError),
        (["300"], TypeError),
        (300.0, TypeError),
    ],
)
def test_ocean_heat_content_rejects_invalid_depth_sequences(depths, error):
    with pytest.raises(error, match="depth"):
        _validated_ohc_depths(depths)


@pytest.mark.parametrize(
    "theta,error",
    [
        (10.0, ValueError),
        (np.empty((2, 0)), ValueError),
        (["warm", "cold"], TypeError),
        (np.array([1.0 + 2.0j, 2.0]), TypeError),
    ],
)
def test_ocean_heat_content_rejects_invalid_temperature(theta, error):
    with pytest.raises(error, match="theta"):
        ocean_heat_content(theta, [0.0, 1.0, 2.0], max_depth=1.0)


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"rho0": 0.0}, ValueError, "rho0"),
        ({"rho0": np.nan}, ValueError, "rho0"),
        ({"heat_capacity": -1.0}, ValueError, "heat_capacity"),
        ({"heat_capacity": "3991"}, TypeError, "heat_capacity"),
        ({"require_full_depth": 1}, TypeError, "require_full_depth"),
    ],
)
def test_ocean_heat_content_rejects_invalid_constants_and_policy(
    kwargs,
    error,
    match,
):
    with pytest.raises(error, match=match):
        ocean_heat_content(
            [10.0, 10.0],
            [0.0, 100.0, 200.0],
            max_depth=100.0,
            **kwargs,
        )
