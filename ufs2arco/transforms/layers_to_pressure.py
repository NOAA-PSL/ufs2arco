"""Diagnose selected pressure-level fields from UFS-Replay hybrid layers."""

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

from ufs2arco.layers2pressure import Layers2Pressure


_SUPPORTED_UNITS = {
    "pressure": {"pa", "pascal", "pascals"},
    "temperature": {"k", "kelvin"},
    "humidity": {"kg/kg", "kgkg-1", "kgkg**-1", "1"},
    "thickness": {"m", "meter", "meters", "metre", "metres"},
}

_MAX_FALLBACK_DISTANCE_PA = 1_000.0
_MAX_BRIDGE_SPAN_PA = 2_500.0
_PAIR_DIMENSION = "_layers_to_pressure_pair"


def _normalized_units(value: str) -> str:
    return value.strip().lower().replace(" ", "")


def _require_units(xda: xr.DataArray, kind: str) -> None:
    units = xda.attrs.get("units")
    if not isinstance(units, str) or _normalized_units(units) not in _SUPPORTED_UNITS[kind]:
        expected = sorted(_SUPPORTED_UNITS[kind])
        raise ValueError(
            f"layers_to_pressure: {xda.name!r} must declare {kind} units "
            f"compatible with {expected}, got {units!r}"
        )


def _validate_levels(levels_hpa: Sequence[float]) -> np.ndarray:
    if isinstance(levels_hpa, (str, bytes)) or not isinstance(levels_hpa, Sequence):
        raise TypeError("layers_to_pressure: levels_hpa must be a sequence")

    values = np.asarray(levels_hpa)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("layers_to_pressure: levels_hpa must be a non-empty 1D sequence")
    if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError("layers_to_pressure: levels_hpa must contain finite numbers")
    if np.any(values <= 0) or np.any(values > 1100):
        raise ValueError(
            "layers_to_pressure: levels_hpa must be positive pressure levels in hPa"
        )
    if not np.all(values == values.astype(np.int64)):
        raise ValueError("layers_to_pressure: levels_hpa must contain integer hPa levels")

    values = values.astype(np.int64)
    if np.unique(values).size != values.size:
        raise ValueError("layers_to_pressure: levels_hpa must not contain duplicates")
    return values


def _validate_variables(xds: xr.Dataset, variables: Mapping[str, str], level_name: str) -> None:
    if not isinstance(variables, Mapping) or len(variables) == 0:
        raise TypeError("layers_to_pressure: variables must be a non-empty mapping")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in variables.items()):
        raise TypeError("layers_to_pressure: variable names must be strings")
    if len(set(variables.values())) != len(variables):
        raise ValueError("layers_to_pressure: output variable names must be unique")

    missing = [name for name in variables if name not in xds]
    if missing:
        raise KeyError(f"layers_to_pressure: missing input variables {missing}")

    collisions = [name for name in variables.values() if name in xds.variables]
    if collisions:
        raise ValueError(
            f"layers_to_pressure: output variable names already exist: {collisions}"
        )

    for name in variables:
        if level_name not in xds[name].dims:
            raise ValueError(
                f"layers_to_pressure: {name!r} must have dimension {level_name!r}"
            )
        if not np.issubdtype(xds[name].dtype, np.floating):
            raise TypeError(f"layers_to_pressure: {name!r} must have floating-point data")
        if not isinstance(xds[name].attrs.get("units"), str):
            raise ValueError(f"layers_to_pressure: {name!r} must declare units")


def _validate_replay_grid(
    xds: xr.Dataset,
    converter: Layers2Pressure,
    level_name: str,
) -> None:
    if level_name not in xds.dims or level_name not in xds.coords:
        raise ValueError(
            f"layers_to_pressure: dataset must have coordinate {level_name!r}"
        )

    actual = np.asarray(xds[level_name].values)
    expected = np.asarray(converter.pfull.values)
    if actual.ndim != 1 or actual.size != expected.size:
        raise ValueError(
            "layers_to_pressure: expected the complete 127-level UFS-Replay "
            f"vertical grid, got {actual.size} levels"
        )
    if not np.allclose(actual, expected, rtol=0.0, atol=5e-6):
        raise ValueError(
            "layers_to_pressure: level coordinate does not match the complete, "
            "ordered UFS-Replay 127-level grid"
        )


def _gather_dimension(
    xda: xr.DataArray,
    dimension: str,
    index: xr.DataArray,
) -> xr.DataArray:
    """Gather one positional index per column without retaining its coordinate."""

    position = xr.DataArray(
        np.arange(xda.sizes[dimension]),
        coords={dimension: xda[dimension]},
        dims=dimension,
    )
    selected = position == index.reset_coords(drop=True)
    return xda.where(selected, 0).sum(dimension, skipna=False)


def _adjacent_pairs(
    xda: xr.DataArray,
    level_name: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return positionally aligned neighboring native levels."""

    pair_coordinate = np.arange(xda.sizes[level_name] - 1)
    lower = xda.isel({level_name: slice(None, -1)}).rename(
        {level_name: _PAIR_DIMENSION}
    )
    upper = xda.isel({level_name: slice(1, None)}).rename(
        {level_name: _PAIR_DIMENSION}
    )
    lower = lower.assign_coords({_PAIR_DIMENSION: pair_coordinate})
    upper = upper.assign_coords({_PAIR_DIMENSION: pair_coordinate})
    return lower, upper


def _log_pressure_value(
    pstar: float,
    left_pressure: xr.DataArray,
    right_pressure: xr.DataArray,
    left_value: xr.DataArray,
    right_value: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Interpolate or extrapolate between two ordered pressure levels."""

    denominator = np.log(right_pressure) - np.log(left_pressure)
    factor = (np.log(pstar) - np.log(left_pressure)) / denominator
    valid = (
        np.isfinite(left_pressure)
        & np.isfinite(right_pressure)
        & (left_pressure > 0)
        & (right_pressure > left_pressure)
        & np.isfinite(left_value)
        & np.isfinite(right_value)
        & np.isfinite(factor)
    )
    value = left_value + factor * (right_value - left_value)
    return value, valid


def _interpolate_with_replay_fallback(
    converter: Layers2Pressure,
    source: xr.DataArray,
    pstar: float,
    prsl: xr.DataArray,
) -> xr.DataArray:
    """Interpolate with bounded recovery for an unavailable native level."""

    level_name = converter.level_name
    available_pressure = prsl.where(np.isfinite(source))
    coefficients = converter.get_interp_coefficients(
        pstar,
        available_pressure,
        include_masks=False,
    )

    normal = coefficients["exact"] | (
        coefficients["valid"]
        & (coefficients["right_index"] == coefficients["left_index"] + 1)
    )
    normal_left = _gather_dimension(
        source,
        level_name,
        coefficients["left_index"],
    )
    normal_right = _gather_dimension(
        source,
        level_name,
        coefficients["right_index"],
    )
    result = normal_left + (
        normal_right - normal_left
    ) * coefficients["factor"]
    result = result.where(
        normal & np.isfinite(normal_left) & np.isfinite(normal_right)
    )

    selected_inversion = (
        np.isfinite(coefficients["left_pressure"])
        & np.isfinite(coefficients["right_pressure"])
        & (coefficients["left_pressure"] < pstar)
        & (coefficients["right_pressure"] > pstar)
        & (coefficients["right_index"] <= coefficients["left_index"])
    )
    blocked = selected_inversion

    bridge_position = (
        coefficients["right_index"] == coefficients["left_index"] + 2
    )
    bridge_bracket = (
        np.isfinite(coefficients["left_pressure"])
        & np.isfinite(coefficients["right_pressure"])
        & (coefficients["left_pressure"] > 0)
        & (coefficients["left_pressure"] < pstar)
        & (coefficients["right_pressure"] > pstar)
    )
    middle_index = (coefficients["left_index"] + 1).clip(
        max=source.sizes[level_name] - 1
    )
    middle_pressure = _gather_dimension(prsl, level_name, middle_index)
    middle_value = _gather_dimension(source, level_name, middle_index)
    finite_middle_pressure = np.isfinite(middle_pressure) & (middle_pressure > 0)
    ordered_middle = (
        (coefficients["left_pressure"] < middle_pressure)
        & (middle_pressure < coefficients["right_pressure"])
    )
    blocked = blocked | (
        bridge_position
        & bridge_bracket
        & finite_middle_pressure
        & ~ordered_middle
    )
    middle_available = finite_middle_pressure & np.isfinite(middle_value)

    bridge_value, bridge_valid = _log_pressure_value(
        pstar,
        coefficients["left_pressure"],
        coefficients["right_pressure"],
        _gather_dimension(source, level_name, coefficients["left_index"]),
        _gather_dimension(source, level_name, coefficients["right_index"]),
    )
    bridge_valid = (
        bridge_valid
        & bridge_position
        & bridge_bracket
        & ~middle_available
        & (~finite_middle_pressure | ordered_middle)
        & (
            coefficients["right_pressure"] - coefficients["left_pressure"]
            <= _MAX_BRIDGE_SPAN_PA
        )
    )
    use_bridge = ~np.isfinite(result) & ~blocked & bridge_valid
    result = xr.where(use_bridge, bridge_value, result)

    left_pressure, right_pressure = _adjacent_pairs(prsl, level_name)
    left_value, right_value = _adjacent_pairs(source, level_name)
    below_target = right_pressure < pstar
    endpoint_distance = xr.where(
        below_target,
        pstar - right_pressure,
        np.inf,
    )
    pair_span = right_pressure - left_pressure
    extrapolation_candidates = (
        np.isfinite(left_pressure)
        & np.isfinite(right_pressure)
        & np.isfinite(left_value)
        & np.isfinite(right_value)
        & (left_pressure > 0)
        & (right_pressure > left_pressure)
        & below_target
        & (endpoint_distance > 0)
        & (endpoint_distance <= _MAX_FALLBACK_DISTANCE_PA)
        & (endpoint_distance <= pair_span)
    )
    has_extrapolation = extrapolation_candidates.any(_PAIR_DIMENSION)
    extrapolation_index = endpoint_distance.where(
        extrapolation_candidates,
        np.inf,
    ).argmin(_PAIR_DIMENSION).astype(np.int64)
    extrapolation_left_pressure = _gather_dimension(
        left_pressure,
        _PAIR_DIMENSION,
        extrapolation_index,
    )
    extrapolation_right_pressure = _gather_dimension(
        right_pressure,
        _PAIR_DIMENSION,
        extrapolation_index,
    )
    extrapolation_value, extrapolation_valid = _log_pressure_value(
        pstar,
        extrapolation_left_pressure,
        extrapolation_right_pressure,
        _gather_dimension(left_value, _PAIR_DIMENSION, extrapolation_index),
        _gather_dimension(right_value, _PAIR_DIMENSION, extrapolation_index),
    )
    use_extrapolation = (
        ~np.isfinite(result)
        & ~blocked
        & has_extrapolation
        & extrapolation_valid
    )
    result = xr.where(use_extrapolation, extrapolation_value, result)

    distance = np.abs(available_pressure - pstar)
    nearest_candidates = (
        np.isfinite(available_pressure)
        & (available_pressure > 0)
        & (available_pressure < pstar)
        & (distance <= _MAX_FALLBACK_DISTANCE_PA)
    )
    one_nearest_candidate = nearest_candidates.sum(level_name) == 1
    nearest_index = distance.where(nearest_candidates, np.inf).argmin(
        level_name
    ).astype(np.int64)
    nearest_value = _gather_dimension(source, level_name, nearest_index)
    pair_index = xr.DataArray(
        np.arange(source.sizes[level_name] - 1),
        coords={_PAIR_DIMENSION: left_pressure[_PAIR_DIMENSION]},
        dims=_PAIR_DIMENSION,
    )
    inverted_pair = (
        np.isfinite(left_pressure)
        & np.isfinite(right_pressure)
        & (left_pressure > 0)
        & (right_pressure > 0)
        & (right_pressure < left_pressure)
    )
    nearest_touches_inversion = (
        inverted_pair
        & (
            (pair_index == nearest_index)
            | (pair_index + 1 == nearest_index)
        )
    ).any(_PAIR_DIMENSION)
    use_nearest = (
        ~np.isfinite(result)
        & ~blocked
        & one_nearest_candidate
        & ~nearest_touches_inversion
        & np.isfinite(nearest_value)
    )
    return xr.where(use_nearest, nearest_value, result)


def layers_to_pressure(
    xds: xr.Dataset,
    variables: Mapping[str, str],
    levels_hpa: Sequence[float],
    pressure_var: str = "pressfc",
    temperature_var: str = "tmp",
    humidity_var: str = "spfh",
    thickness_var: str = "delz",
    below_ground: str = "mask",
    level_name: str = "level",
    interface_name: str = "phalf",
) -> xr.Dataset:
    """Interpolate full-column UFS-Replay fields to actual pressure levels.

    All 127 native hybrid layers are required. Pressure is diagnosed from the
    native surface pressure, temperature, humidity, and layer thickness. The
    nearest pressure on each side of every target is gathered per column and
    the requested field is interpolated linearly in log-pressure. If one native
    level is unavailable, the transform can bridge that level, extrapolate from
    an adjacent pair, or use one nearest value within 10 hPa. Columns that are
    below ground or otherwise invalid remain missing.
    """

    if not isinstance(xds, xr.Dataset):
        raise TypeError("layers_to_pressure: xds must be an xarray.Dataset")
    if below_ground != "mask":
        raise ValueError("layers_to_pressure: only below_ground='mask' is supported")

    levels = _validate_levels(levels_hpa)
    _validate_variables(xds, variables, level_name)

    support_variables = [
        pressure_var,
        temperature_var,
        humidity_var,
        thickness_var,
    ]
    missing_support = [name for name in support_variables if name not in xds]
    if missing_support:
        raise KeyError(f"layers_to_pressure: missing support variables {missing_support}")

    for name in (temperature_var, humidity_var, thickness_var):
        if level_name not in xds[name].dims:
            raise ValueError(
                f"layers_to_pressure: support variable {name!r} must have "
                f"dimension {level_name!r}"
            )
    if level_name in xds[pressure_var].dims:
        raise ValueError(
            f"layers_to_pressure: surface pressure {pressure_var!r} must not "
            f"have dimension {level_name!r}"
        )
    if not np.issubdtype(xds[pressure_var].dtype, np.floating):
        raise TypeError(
            f"layers_to_pressure: surface pressure {pressure_var!r} must be "
            "floating point"
        )

    _require_units(xds[pressure_var], "pressure")
    _require_units(xds[temperature_var], "temperature")
    _require_units(xds[humidity_var], "humidity")
    _require_units(xds[thickness_var], "thickness")
    invalid_thickness = (
        np.isfinite(xds[thickness_var]) & (xds[thickness_var] >= 0)
    ).any()
    if bool(invalid_thickness.values):
        raise ValueError(
            f"layers_to_pressure: finite {thickness_var!r} values must be "
            "negative layer thicknesses in metres"
        )

    converter = Layers2Pressure(
        level_name=level_name,
        interface_name=interface_name,
    )
    _validate_replay_grid(xds, converter, level_name)

    prsl = converter.calc_layer_mean_pressure(
        pressfc=xds[pressure_var],
        temp=xds[temperature_var],
        spfh=xds[humidity_var],
        delz=xds[thickness_var],
    )

    interpolated_by_output = {output_name: [] for output_name in variables.values()}
    for level in levels:
        level_hpa = int(level)
        level_pa = float(level_hpa) * 100.0
        for source_name, output_name in variables.items():
            source = xds[source_name]
            interpolated = _interpolate_with_replay_fallback(
                converter,
                source,
                level_pa,
                prsl,
            )
            valid_surface = np.isfinite(xds[pressure_var]) & (xds[pressure_var] > level_pa)
            interpolated = interpolated.where(valid_surface).astype(source.dtype)
            interpolated_by_output[output_name].append(
                interpolated.expand_dims({level_name: [level_hpa]})
            )

    output_variables = {}
    for source_name, output_name in variables.items():
        source = xds[source_name]
        output = xr.concat(
            interpolated_by_output[output_name],
            dim=level_name,
        ).transpose(*source.dims).rename(output_name)
        output.attrs = source.attrs.copy()
        derived_from = list(dict.fromkeys([*support_variables, source_name]))
        output.attrs.update(
            {
                "source_variable": source_name,
                "vertical_interpolation": "linear_in_log_pressure",
                "source_vertical_grid": "ufs_replay_127_layer_hybrid",
                "support_variables": support_variables.copy(),
                "required_support_fields": support_variables.copy(),
                "derived_from": derived_from,
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
                "type_of_level": "isobaricInhPa",
            }
        )
        output_variables[output_name] = output

    result = xr.Dataset(output_variables, attrs=xds.attrs.copy())
    result[level_name].attrs.update(
        {
            "units": "hPa",
            "standard_name": "air_pressure",
            "long_name": "pressure level",
            "type_of_level": "isobaricInhPa",
        }
    )
    return result
