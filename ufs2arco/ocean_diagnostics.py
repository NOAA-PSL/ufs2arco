"""Memory-bounded ocean diagnostics."""

import logging
from numbers import Real
from typing import Any

import numpy as np

logger = logging.getLogger("ufs2arco")

# Wright (1997) coefficients transcribed from the MOM6 full- and reduced-range
# equations of state. Coefficient order is
# (a0, a1, a2, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5).
_WRIGHT_COEFFICIENTS = {
    "wright_full": (
        7.133718e-4,
        2.724670e-7,
        -1.646582e-7,
        5.613770e8,
        3.600337e6,
        -3.727194e4,
        1.660557e2,
        6.844158e5,
        -8.389457e3,
        1.609893e5,
        8.427815e2,
        -6.931554,
        3.869318e-2,
        -1.664201e2,
        -2.765195,
    ),
    "wright_red": (
        7.057924e-4,
        3.480336e-7,
        -1.112733e-7,
        5.790749e8,
        3.516535e6,
        -4.002714e4,
        2.084372e2,
        5.944068e5,
        -9.643486e3,
        1.704853e5,
        7.904722e2,
        -7.984422,
        5.140652e-2,
        -2.302158e2,
        -3.079464,
    ),
}

# MOM6 fit bounds: temperature (degC), salinity (psu), and pressure (Pa).
_WRIGHT_VALID_RANGE = {
    "wright_full": (-2.0, 40.0, 0.0, 40.0, 0.0, 1.0e8),
    "wright_red": (-2.0, 30.0, 28.0, 38.0, 0.0, 5.0e7),
}

DEFAULT_EOS = "wright_full"

# MOM6 Boussinesq defaults: kg m-3, J kg-1 K-1, and m s-2, respectively.
RHO_0 = 1035.0
HEAT_CAPACITY = 3991.86795711963
GRAVITY = 9.8


def _validated_positive_constant(name: str, value: float) -> float:
    """Return a finite positive real-valued constant."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validated_real_array(name: str, values: np.ndarray) -> np.ndarray:
    """Return a real numeric array as float64 without modifying its input."""
    array = np.asarray(values)
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must contain real numeric data")
    return np.asarray(array, dtype=np.float64)


def hydrostatic_pressure(
    depth: np.ndarray | float,
    rho0: float = RHO_0,
    gravity: float = GRAVITY,
) -> np.ndarray:
    """Return Boussinesq hydrostatic pressure at positive-down depth.

    Args:
        depth: Finite nonnegative depth below the surface in metres.
        rho0: Finite positive reference density in kg m-3.
        gravity: Finite positive gravitational acceleration in m s-2.

    Returns:
        Pressure in Pa as a float64 NumPy array or scalar.
    """
    depth_array = _validated_real_array("depth", depth)
    if not np.all(np.isfinite(depth_array)) or np.any(depth_array < 0.0):
        raise ValueError("depth must be finite and nonnegative")
    density = _validated_positive_constant("rho0", rho0)
    acceleration = _validated_positive_constant("gravity", gravity)
    return density * acceleration * depth_array


def interfaces_from_centers(
    depth: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Reconstruct surface-pinned layer interfaces from midpoint depths.

    The recurrence ``z_i[k + 1] = 2 * z_l[k] - z_i[k]`` is valid for a
    z-coordinate whose layer centers bisect adjacent interfaces. Coordinates
    that do not yield finite, nonnegative, strictly increasing interfaces are
    rejected rather than silently producing invalid cell thicknesses.
    """
    centers = _validated_real_array("layer-center depths", depth)
    if centers.ndim != 1:
        raise ValueError(
            "interfaces_from_centers requires a 1D layer-center coordinate, "
            f"got shape {centers.shape}"
        )
    if centers.size == 0:
        raise ValueError("interfaces_from_centers requires at least one center")
    if not np.all(np.isfinite(centers)):
        raise ValueError("layer-center depths must be finite")
    if np.any(centers < 0.0) or not np.all(np.diff(centers) > 0.0):
        raise ValueError(
            "layer-center depths must be nonnegative and strictly increasing"
        )

    interfaces = np.empty(centers.size + 1, dtype=np.float64)
    interfaces[0] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for index, center in enumerate(centers):
            interfaces[index + 1] = 2.0 * center - interfaces[index]

    if (
        not np.all(np.isfinite(interfaces))
        or np.any(interfaces < 0.0)
        or not np.all(np.diff(interfaces) > 0.0)
    ):
        raise ValueError(
            "reconstructed interfaces are not finite, nonnegative, and "
            "strictly increasing; pass explicit interfaces for this grid"
        )
    return interfaces


def _validated_interfaces(
    interfaces: np.ndarray | list[float] | tuple[float, ...],
    level_count: int,
) -> np.ndarray:
    """Return validated surface-starting interfaces as a float64 array."""
    if isinstance(level_count, (bool, np.bool_)) or not isinstance(
        level_count, (int, np.integer)
    ):
        raise TypeError("level_count must be an integer")
    if level_count < 1:
        raise ValueError("level_count must be positive")

    values = _validated_real_array("interfaces", interfaces)
    if values.ndim != 1:
        raise ValueError(f"interfaces must be 1D, got shape {values.shape}")
    expected = int(level_count) + 1
    if values.size != expected:
        raise ValueError(
            f"got {values.size} interfaces for {level_count} levels; "
            f"expected {expected}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("interfaces must be finite")
    if values[0] != 0.0:
        raise ValueError("interfaces must start at the ocean surface (0 m)")
    if np.any(values < 0.0) or not np.all(np.diff(values) > 0.0):
        raise ValueError(
            "interfaces must be nonnegative and strictly increasing with depth"
        )
    return values


def _validated_ohc_depths(
    max_depths: list[float | None] | tuple[float | None, ...] | np.ndarray,
) -> tuple[float | None, ...]:
    """Return unique finite positive OHC depths, preserving ``None``."""
    if isinstance(max_depths, np.ndarray):
        if max_depths.ndim != 1:
            raise ValueError("ocean heat content depths must be a 1D sequence")
        candidates = max_depths.tolist()
    elif isinstance(max_depths, (list, tuple)):
        candidates = list(max_depths)
    else:
        raise TypeError("ocean heat content depths must be a sequence")
    if not candidates:
        raise ValueError("ocean heat content depths must not be empty")

    validated: list[float | None] = []
    for depth in candidates:
        if depth is None:
            value = None
        else:
            if isinstance(depth, (bool, np.bool_)) or not isinstance(depth, Real):
                raise TypeError(
                    "ocean heat content depths must contain real numbers or None"
                )
            value = float(depth)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "ocean heat content depths must be finite and positive"
                )
        if value in validated:
            raise ValueError("ocean heat content depths must be unique")
        validated.append(value)
    return tuple(validated)


def available_eos() -> tuple[str, ...]:
    """Return the implemented equation-of-state names."""
    return tuple(_WRIGHT_COEFFICIENTS)


def _validate_eos(eos: str) -> str:
    """Return a validated equation-of-state name."""
    if eos not in _WRIGHT_COEFFICIENTS:
        raise NotImplementedError(
            f"unrecognized equation of state {eos!r}; implemented options are "
            f"{available_eos()}"
        )
    return eos


def _as_float64_operand(value: Any) -> Any:
    """Cast an array-like operand without forcing lazy arrays to compute."""
    astype = getattr(value, "astype", None)
    if callable(astype):
        return astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def _warn_outside_range(theta: Any, salt: Any, pressure: Any, eos: str) -> None:
    """Warn when input ranges exceed the selected Wright fit's bounds."""
    tmin, tmax, smin, smax, pmin, pmax = _WRIGHT_VALID_RANGE[eos]
    for name, values, lower, upper in (
        ("potential temperature", theta, tmin, tmax),
        ("salinity", salt, smin, smax),
        ("pressure", pressure, pmin, pmax),
    ):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).any():
            continue
        low, high = np.nanmin(values), np.nanmax(values)
        if low < lower or high > upper:
            logger.warning(
                "wright_density: %s range [%.4g, %.4g] falls outside "
                "the %s fit range [%g, %g]; densities there are extrapolated",
                name,
                low,
                high,
                eos,
                lower,
                upper,
            )


def wright_density(
    theta: Any,
    salt: Any,
    pressure: Any,
    eos: str = DEFAULT_EOS,
    check_range: bool = False,
) -> Any:
    """Calculate seawater density using the Wright (1997) MOM6 expression.

    Pressure is in Pa. Passing zero computes potential density referenced to
    the surface, which is the density used by the MOM6 mixed-layer-depth
    diagnostic. Setting ``check_range=True`` logs warnings for inputs outside
    the selected fit's range and evaluates lazy inputs to inspect their bounds.
    """
    eos = _validate_eos(eos)
    if check_range:
        _warn_outside_range(theta, salt, pressure, eos)
    theta = _as_float64_operand(theta)
    salt = _as_float64_operand(salt)
    pressure = _as_float64_operand(pressure)
    (
        a0,
        a1,
        a2,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
    ) = _WRIGHT_COEFFICIENTS[eos]

    al0 = a0 + (a1 * theta + a2 * salt)
    p0 = b0 + (
        b4 * salt + theta * (b1 + (theta * (b2 + b3 * theta) + b5 * salt))
    )
    lam = c0 + (
        c4 * salt + theta * (c1 + (theta * (c2 + c3 * theta) + c5 * salt))
    )
    return (pressure + p0) / (lam + al0 * (pressure + p0))


def _validated_thresholds(
    thresholds: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Return finite positive density thresholds as a 1D float64 array."""
    values = np.asarray(thresholds, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("mixed layer depth thresholds must be a non-empty 1D sequence")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("mixed layer depth thresholds must be finite and positive")
    return values


def _validated_depth(
    depth: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Return validated positive-down layer-center depths in metres."""
    values = np.asarray(depth, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"mixed layer depth requires a 1D depth coordinate, got shape {values.shape}"
        )
    if values.size < 2:
        raise ValueError(
            f"mixed layer depth requires at least two vertical levels, got {values.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("mixed layer depth requires finite layer-center depths")
    if np.any(values < 0.0) or not np.all(np.diff(values) > 0.0):
        raise ValueError(
            "mixed layer depth requires nonnegative layer-center depths that "
            "strictly increase downward"
        )
    return values


def _validate_ref_pressure(ref_pressure: float) -> float:
    """Require the MOM6 default surface reference pressure."""
    try:
        value = float(ref_pressure)
    except (TypeError, ValueError) as error:
        raise NotImplementedError(
            "mixed layer depth currently supports only ref_pressure=0"
        ) from error
    if not np.isfinite(value) or value != 0.0:
        raise NotImplementedError(
            "mixed layer depth currently supports only ref_pressure=0; "
            f"got {ref_pressure!r}"
        )
    return value


def _mixed_layer_depth_for_thresholds(
    theta: np.ndarray,
    salt: np.ndarray,
    depth: np.ndarray | list[float] | tuple[float, ...],
    thresholds: np.ndarray | list[float] | tuple[float, ...],
    ref_pressure: float = 0.0,
    eos: str = DEFAULT_EOS,
    pathological_to_nan: bool = True,
) -> np.ndarray:
    """Compute several density-threshold MLD fields in one vertical pass.

    The final axis of ``theta`` and ``salt`` is vertical. Work arrays are
    horizontal slices rather than full float64 columns, which bounds temporary
    memory while retaining MOM6's first rising threshold-crossing rule.
    """
    level = _validated_depth(depth)
    density_thresholds = _validated_thresholds(thresholds)
    reference_pressure = _validate_ref_pressure(ref_pressure)
    eos = _validate_eos(eos)
    if not isinstance(pathological_to_nan, (bool, np.bool_)):
        raise TypeError("pathological_to_nan must be a boolean")

    theta_array = np.asarray(theta)
    salt_array = np.asarray(salt)
    for name, array in (("theta", theta_array), ("salt", salt_array)):
        if array.ndim == 0:
            raise ValueError(f"{name} must have a vertical axis")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain numeric data")
        if array.shape[-1] != level.size:
            raise ValueError(
                f"{name} has {array.shape[-1]} vertical levels but depth has "
                f"{level.size}"
            )

    try:
        theta_array, salt_array = np.broadcast_arrays(theta_array, salt_array)
    except ValueError as error:
        raise ValueError(
            f"theta and salt shapes are not broadcast-compatible: "
            f"{theta_array.shape} and {salt_array.shape}"
        ) from error

    leading_shape = theta_array.shape[:-1]
    threshold_shape = (1,) * len(leading_shape) + (density_thresholds.size,)
    broadcast_thresholds = density_thresholds.reshape(threshold_shape)

    surface_theta = np.asarray(theta_array[..., 0], dtype=np.float64)
    surface_salt = np.asarray(salt_array[..., 0], dtype=np.float64)
    active = np.isfinite(surface_theta) & np.isfinite(surface_salt)
    is_ocean = active.copy()
    surface_density = wright_density(
        surface_theta,
        surface_salt,
        reference_pressure,
        eos=eos,
    )

    previous_delta = np.zeros(leading_shape, dtype=np.float64)
    deepest_delta = previous_delta.copy()
    deepest_depth = np.full(leading_shape, level[0], dtype=np.float64)
    found = np.zeros(leading_shape + (density_thresholds.size,), dtype=bool)
    result = np.zeros(leading_shape + (density_thresholds.size,), dtype=np.float64)

    for index in range(1, level.size):
        theta_slice = np.asarray(theta_array[..., index], dtype=np.float64)
        salt_slice = np.asarray(salt_array[..., index], dtype=np.float64)
        active &= np.isfinite(theta_slice) & np.isfinite(salt_slice)

        density = wright_density(
            theta_slice,
            salt_slice,
            reference_pressure,
            eos=eos,
        )
        current_delta = density - surface_density
        rise = current_delta - previous_delta

        crossing = (
            active[..., None]
            & ~found
            & (rise[..., None] > 0.0)
            & (previous_delta[..., None] < broadcast_thresholds)
            & (current_delta[..., None] >= broadcast_thresholds)
        )
        safe_rise = np.where(crossing, rise[..., None], 1.0)
        fraction = (
            broadcast_thresholds - previous_delta[..., None]
        ) / safe_rise
        interpolated = (
            fraction * level[index] + (1.0 - fraction) * level[index - 1]
        )
        result = np.where(crossing, interpolated, result)
        found |= crossing

        previous_delta = np.where(active, current_delta, previous_delta)
        deepest_delta = np.where(active, current_delta, deepest_delta)
        deepest_depth = np.where(active, level[index], deepest_depth)

    reaches_bottom = (
        ~found
        & is_ocean[..., None]
        & (deepest_delta[..., None] < broadcast_thresholds)
    )
    result = np.where(reaches_bottom, deepest_depth[..., None], result)

    pathological = ~found & ~reaches_bottom & is_ocean[..., None]
    for threshold_index, threshold in enumerate(density_thresholds):
        count = int(np.count_nonzero(pathological[..., threshold_index]))
        if count == 0:
            continue
        disposition = (
            "set to NaN"
            if pathological_to_nan
            else "left at 0.0, as MOM6 does"
        )
        logger.warning(
            "mixed_layer_depth_by_density_difference: %d ocean column(s) "
            "resolved to neither a threshold crossing nor the bottom for "
            "threshold %.6g kg m-3; %s",
            count,
            threshold,
            disposition,
        )

    if pathological_to_nan:
        result = np.where(pathological, np.nan, result)
    return np.where(is_ocean[..., None], result, np.nan)


def mixed_layer_depth_by_density_difference(
    theta: np.ndarray,
    salt: np.ndarray,
    depth: np.ndarray | list[float] | tuple[float, ...],
    density_diff: float = 0.03,
    ref_pressure: float = 0.0,
    eos: str = DEFAULT_EOS,
    pathological_to_nan: bool = True,
) -> np.ndarray:
    """Compute MOM6-style mixed layer depth for one density threshold.

    ``theta`` and ``salt`` must have matching or broadcast-compatible shapes;
    their final axis is the native vertical column. Temperature is potential
    temperature in degrees Celsius, salinity is practical salinity in psu,
    ``depth`` contains positive-down nominal layer-center depths in metres, and
    ``density_diff`` is in kg m-3. The returned float64 array has the broadcast
    input shape with the final vertical axis removed.

    The first downward crossing satisfying
    ``previous < density_diff <= current`` and a positive density rise is
    linearly interpolated between layer centers. A wholly mixed valid column
    falls back to its deepest contiguous valid layer center; a missing surface
    is land and returns NaN. The first jointly non-finite temperature/salinity
    level terminates the valid water column. This nominal-center geometry does
    not reconstruct time-varying MOM6 ALE layer positions from ``ho``.

    Raises:
        ValueError: If shapes, depths, or the density threshold are invalid.
        TypeError: If inputs are nonnumeric or policies have invalid types.
        NotImplementedError: If the EOS is unknown or reference pressure is
            not zero.
    """
    result = _mixed_layer_depth_for_thresholds(
        theta,
        salt,
        depth=depth,
        thresholds=[density_diff],
        ref_pressure=ref_pressure,
        eos=eos,
        pathological_to_nan=pathological_to_nan,
    )
    return result[..., 0]


def _ocean_heat_content_for_depths(
    theta: np.ndarray,
    interfaces: np.ndarray | list[float] | tuple[float, ...],
    max_depths: list[float | None] | tuple[float | None, ...] | np.ndarray,
    rho0: float = RHO_0,
    heat_capacity: float = HEAT_CAPACITY,
    require_full_depth: bool = True,
) -> np.ndarray:
    """Integrate several heat-content depths in one vertical scan.

    The final axis of ``theta`` is vertical. Temporary arrays are horizontal
    slices, plus one horizontal field per requested depth, so memory remains
    bounded independently of the number of vertical levels. Valid water must
    form an unbroken run from the surface; the first non-finite value marks the
    local bottom even if deeper values are finite.

    ``None`` requests the whole contiguous valid column. Finite depths use
    exact finite-volume partial-cell weighting. Results are always float64 in
    J m-2, with the requested-depth axis appended last.
    """
    theta_array = np.asarray(theta)
    if theta_array.ndim == 0:
        raise ValueError("theta must have a final vertical axis")
    if (
        not np.issubdtype(theta_array.dtype, np.number)
        or np.issubdtype(theta_array.dtype, np.complexfloating)
    ):
        raise TypeError("theta must contain real numeric data")
    level_count = theta_array.shape[-1]
    if level_count < 1:
        raise ValueError("theta must contain at least one vertical level")

    interface_values = _validated_interfaces(interfaces, level_count)
    target_depths = _validated_ohc_depths(max_depths)
    density = _validated_positive_constant("rho0", rho0)
    capacity = _validated_positive_constant("heat_capacity", heat_capacity)
    if not isinstance(require_full_depth, (bool, np.bool_)):
        raise TypeError("require_full_depth must be a boolean")

    leading_shape = theta_array.shape[:-1]
    result = np.zeros(
        leading_shape + (len(target_depths),),
        dtype=np.float64,
    )
    active = np.ones(leading_shape, dtype=bool)
    is_ocean = np.zeros(leading_shape, dtype=bool)
    column_bottom = np.zeros(leading_shape, dtype=np.float64)

    for level_index in range(level_count):
        theta_slice = np.asarray(theta_array[..., level_index], dtype=np.float64)
        active &= np.isfinite(theta_slice)
        if level_index == 0:
            is_ocean = active.copy()

        lower = interface_values[level_index]
        upper = interface_values[level_index + 1]
        thickness = upper - lower
        valid_theta = np.where(active, theta_slice, 0.0)
        column_bottom = np.where(active, upper, column_bottom)

        for depth_index, target_depth in enumerate(target_depths):
            if target_depth is None:
                included_thickness = thickness
            else:
                included_thickness = np.clip(
                    target_depth - lower,
                    0.0,
                    thickness,
                )
            if included_thickness > 0.0:
                result[..., depth_index] += valid_theta * included_thickness

    result *= density * capacity
    for depth_index, target_depth in enumerate(target_depths):
        valid_result = is_ocean
        if target_depth is not None and require_full_depth:
            valid_result = valid_result & (column_bottom >= target_depth)
        result[..., depth_index] = np.where(
            valid_result,
            result[..., depth_index],
            np.nan,
        )
    return result


def ocean_heat_content(
    theta: np.ndarray,
    interfaces: np.ndarray | list[float] | tuple[float, ...],
    max_depth: float | None = None,
    rho0: float = RHO_0,
    heat_capacity: float = HEAT_CAPACITY,
    require_full_depth: bool = True,
) -> np.ndarray:
    """Return depth-integrated ocean heat content in J m-2.

    This evaluates the Boussinesq finite-volume integral
    ``rho0 * heat_capacity * integral(theta dz)`` relative to 0 degrees C.
    ``max_depth`` selects a finite positive integration depth in metres and
    applies partial-cell weighting at that depth. ``None`` integrates the
    whole contiguous valid water column. When ``require_full_depth`` is true,
    finite-depth results are NaN where the local water column is shallower than
    the requested depth.

    The final axis of ``theta`` must correspond to the layers bounded by
    ``interfaces``. A missing surface is land and returns NaN; an internal
    non-finite value terminates the valid column.
    """
    result = _ocean_heat_content_for_depths(
        theta,
        interfaces=interfaces,
        max_depths=[max_depth],
        rho0=rho0,
        heat_capacity=heat_capacity,
        require_full_depth=require_full_depth,
    )
    return result[..., 0]
