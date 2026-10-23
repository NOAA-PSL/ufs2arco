"""Dataset-level ocean diagnostic transforms."""

import logging
import re
from collections.abc import Sequence

import numpy as np
import xarray as xr

from ufs2arco.ocean_diagnostics import (
    DEFAULT_EOS,
    GRAVITY,
    HEAT_CAPACITY,
    RHO_0,
    _mixed_layer_depth_for_thresholds,
    _ocean_heat_content_for_depths,
    _validated_interfaces,
    _validated_ohc_depths,
    _validated_positive_constant,
    available_eos,
    hydrostatic_pressure,
    interfaces_from_centers,
    wright_density,
)

logger = logging.getLogger("ufs2arco")

_TEMPERATURE_UNITS = {
    "c",
    "celsius",
    "degc",
    "degreec",
    "degreecelsius",
    "degreesc",
    "degreescelsius",
}
_SALINITY_UNITS = {
    "psu",
    "practicalsalinityunit",
    "practicalsalinityunits",
}
_DEPTH_UNITS = {"m", "meter", "meters", "metre", "metres"}
_NUMERIC_SUFFIX = re.compile(r".+_\d+$")
_JOULES_PER_GIGAJOULE = 1.0e9
_OHC_OUTPUT_UNITS = {
    "jm-2": ("J m-2", 1.0),
    "jm2": ("J m-2", 1.0),
    "gjm-2": ("GJ m-2", 1.0 / _JOULES_PER_GIGAJOULE),
    "gjm2": ("GJ m-2", 1.0 / _JOULES_PER_GIGAJOULE),
}


def _normalized_units(value: str) -> str:
    """Normalize common unit spellings for validation."""
    return (
        value.strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("°", "deg")
    )


def _require_units(
    xda: xr.DataArray,
    valid: set[str],
    description: str,
    caller: str,
) -> None:
    """Require units compatible with a diagnostic input."""
    units = xda.attrs.get("units")
    if not isinstance(units, str) or _normalized_units(units) not in valid:
        raise ValueError(
            f"{caller}: {xda.name!r} must declare {description} units; "
            f"got {units!r}"
        )


def _resolve_eos(eos: str | None, caller: str) -> str:
    """Resolve the user-facing Wright alias."""
    if eos is None:
        return DEFAULT_EOS
    name = str(eos).strip().lower()
    if name == "wright":
        return DEFAULT_EOS
    if name not in available_eos():
        raise NotImplementedError(
            f"{caller}: unrecognized equation of state {eos!r}; "
            f"implemented options are {available_eos() + ('wright',)}"
        )
    return name


def _validate_inputs(
    xds: xr.Dataset,
    temperature: str,
    salinity: str,
    caller: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Validate the temperature and salinity profile fields."""
    if not isinstance(xds, xr.Dataset):
        raise TypeError(f"{caller}: xds must be an xarray.Dataset")
    missing = [name for name in (temperature, salinity) if name not in xds]
    if missing:
        raise KeyError(
            f"{caller}: missing required variables {missing}; "
            f"available data variables are {sorted(xds.data_vars)}"
        )

    temp = xds[temperature]
    salt = xds[salinity]
    for xda in (temp, salt):
        if "level" not in xda.dims:
            raise ValueError(
                f"{caller}: {xda.name!r} must have a 'level' dimension"
            )
        if not np.issubdtype(xda.dtype, np.floating):
            raise TypeError(
                f"{caller}: {xda.name!r} must contain floating-point data"
            )

    if temp.dims != salt.dims or temp.sizes != salt.sizes:
        raise ValueError(
            f"{caller}: temperature and salinity must have identical "
            f"dimensions and sizes; got {temp.sizes} and {salt.sizes}"
        )

    _require_units(temp, _TEMPERATURE_UNITS, "degrees Celsius", caller)
    _require_units(salt, _SALINITY_UNITS, "practical salinity (psu)", caller)
    return temp, salt


def _validate_temperature(
    xds: xr.Dataset,
    temperature: str,
    caller: str,
) -> xr.DataArray:
    """Validate one potential-temperature profile field."""
    if not isinstance(xds, xr.Dataset):
        raise TypeError(f"{caller}: xds must be an xarray.Dataset")
    if temperature not in xds:
        raise KeyError(
            f"{caller}: missing required variable {temperature!r}; "
            f"available data variables are {sorted(xds.data_vars)}"
        )
    temp = xds[temperature]
    if "level" not in temp.dims:
        raise ValueError(
            f"{caller}: {temperature!r} must have a 'level' dimension"
        )
    if not np.issubdtype(temp.dtype, np.floating):
        raise TypeError(
            f"{caller}: {temperature!r} must contain floating-point data"
        )
    _require_units(temp, _TEMPERATURE_UNITS, "degrees Celsius", caller)
    return temp


def _validate_level(
    xds: xr.Dataset,
    caller: str,
    minimum_levels: int = 1,
) -> np.ndarray:
    """Return finite, positive-down layer-center depths in metres."""
    if "level" not in xds.dims or "level" not in xds.coords:
        raise KeyError(
            f"{caller}: dataset must have a 'level' dimension and coordinate"
        )
    level_coord = xds["level"]
    units = level_coord.attrs.get("units")
    if units is not None:
        if not isinstance(units, str) or _normalized_units(units) not in _DEPTH_UNITS:
            raise ValueError(
                f"{caller}: 'level' must be measured in metres when units "
                f"are present; got {units!r}"
            )

    level = np.asarray(level_coord.values, dtype=np.float64)
    if level.ndim != 1:
        raise ValueError(
            f"{caller}: expected a 1D 'level' coordinate, got {level.shape}"
        )
    if level.size < minimum_levels:
        raise ValueError(
            f"{caller}: at least {minimum_levels} vertical level(s) are required, "
            f"got {level.size}"
        )
    if not np.all(np.isfinite(level)):
        raise ValueError(f"{caller}: 'level' must contain only finite depths")
    if np.any(level < 0.0) or not np.all(np.diff(level) > 0.0):
        raise ValueError(
            f"{caller}: 'level' must be nonnegative and strictly increase "
            "with depth"
        )
    return level


def check_vertical_resolution(
    level: np.ndarray,
    mld_search_depth: float = 200.0,
    min_levels_in_search: int = 10,
    max_level_spacing: float = 25.0,
    caller: str = "check_vertical_resolution",
    expected_level_count: int | None = 75,
    minimum_deepest_level: float | None = 5_000.0,
) -> str:
    """Reject an incomplete or upper-ocean grid too coarse to diagnose MLD."""
    if not np.isfinite(mld_search_depth) or mld_search_depth <= 0.0:
        raise ValueError(f"{caller}: mld_search_depth must be finite and positive")
    if (
        isinstance(min_levels_in_search, bool)
        or not isinstance(min_levels_in_search, (int, np.integer))
        or min_levels_in_search < 2
    ):
        raise ValueError(f"{caller}: min_levels_in_search must be an integer >= 2")
    if not np.isfinite(max_level_spacing) or max_level_spacing <= 0.0:
        raise ValueError(f"{caller}: max_level_spacing must be finite and positive")

    if expected_level_count is not None:
        if (
            isinstance(expected_level_count, bool)
            or not isinstance(expected_level_count, (int, np.integer))
            or expected_level_count < 2
        ):
            raise ValueError(
                f"{caller}: expected_level_count must be an integer >= 2 "
                "or null"
            )
        if level.size != int(expected_level_count):
            raise ValueError(
                f"{caller}: found {level.size} vertical levels, but the "
                f"complete Replay column has {expected_level_count}. Ocean MLD "
                "requires the full native water column; remove source.levels or "
                "source.slices.sel.level."
            )

    if minimum_deepest_level is not None:
        if (
            not np.isfinite(minimum_deepest_level)
            or minimum_deepest_level <= 0.0
        ):
            raise ValueError(
                f"{caller}: minimum_deepest_level must be finite and "
                "positive or null"
            )
        if level[-1] < float(minimum_deepest_level):
            raise ValueError(
                f"{caller}: deepest level is {level[-1]:.4g} m, but the "
                f"native Replay grid must extend to at least "
                f"{minimum_deepest_level:g} m so bottom fallback is not changed "
                "by a truncated source column."
            )

    upper_ocean = level[level < float(mld_search_depth)]
    hint = (
        "Ocean MLD requires the native water column; remove source.levels or "
        "source.slices.sel.level and do not use a pre-coarsened source store."
    )
    if upper_ocean.size < int(min_levels_in_search):
        raise ValueError(
            f"{caller}: found {upper_ocean.size} levels above "
            f"{mld_search_depth:g} m, but at least {min_levels_in_search} are "
            f"required. {hint}"
        )

    spacing = np.diff(upper_ocean)
    worst_spacing = float(spacing.max())
    if worst_spacing > float(max_level_spacing):
        index = int(np.argmax(spacing))
        raise ValueError(
            f"{caller}: vertical spacing reaches {worst_spacing:.4g} m "
            f"between {upper_ocean[index]:.4g} and "
            f"{upper_ocean[index + 1]:.4g} m, exceeding the "
            f"{max_level_spacing:g} m limit. {hint}"
        )

    return (
        f"validated complete native grid: {level.size} levels spanning "
        f"{level[0]:.4g}-{level[-1]:.4g} m; {upper_ocean.size} levels above "
        f"{mld_search_depth:g} m, spacing {float(spacing.min()):.4g}-"
        f"{worst_spacing:.4g} m"
    )


def _validate_level_chunks(caller: str, *arrays: xr.DataArray) -> None:
    """Reject split vertical Dask chunks rather than allowing hidden rechunking."""
    for xda in arrays:
        if xda.chunks is None:
            continue
        level_chunks = xda.chunksizes.get("level", ())
        if len(level_chunks) > 1:
            raise ValueError(
                f"{caller}: {xda.name!r} has split 'level' chunks "
                f"{level_chunks}; rechunk the native column with "
                ".chunk({'level': -1}) before applying the transform"
            )


def _as_thresholds(
    thresholds: Sequence[float] | np.ndarray | float,
) -> np.ndarray:
    """Normalize and validate configured density thresholds."""
    if isinstance(thresholds, (str, bytes)):
        raise TypeError("mixed_layer_depth: thresholds must be numeric")
    if isinstance(thresholds, np.ndarray):
        values = np.asarray(thresholds, dtype=np.float64)
    elif isinstance(thresholds, Sequence):
        values = np.asarray(list(thresholds), dtype=np.float64)
    else:
        values = np.asarray([thresholds], dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("mixed_layer_depth: thresholds must be a non-empty 1D sequence")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("mixed_layer_depth: thresholds must be finite and positive")
    if np.unique(values).size != values.size:
        raise ValueError("mixed_layer_depth: thresholds must not contain duplicates")
    return values


def _resolve_names(
    names: Sequence[str] | str | None,
    threshold_count: int,
    xds: xr.Dataset,
) -> list[str]:
    """Validate output names before the dataset is mutated."""
    if names is None:
        if threshold_count != 1:
            raise ValueError(
                "mixed_layer_depth: multiple thresholds require one explicit "
                "output name per threshold"
            )
        output_names = ["mld"]
    elif isinstance(names, str):
        output_names = [names]
    elif not isinstance(names, Sequence):
        raise TypeError(
            "mixed_layer_depth: names must be a string or sequence of strings"
        )
    else:
        output_names = list(names)

    return _validate_output_names(
        output_names,
        expected_count=threshold_count,
        xds=xds,
        caller="mixed_layer_depth",
        value_description="thresholds",
    )


def _validate_output_names(
    output_names: Sequence[str],
    expected_count: int,
    xds: xr.Dataset,
    caller: str,
    value_description: str,
) -> list[str]:
    """Validate derived channel names before mutating the dataset."""
    output_names = list(output_names)
    if len(output_names) != expected_count:
        raise ValueError(
            f"{caller}: got {len(output_names)} output names for "
            f"{expected_count} {value_description}"
        )
    if not all(isinstance(name, str) and name.strip() for name in output_names):
        raise TypeError(f"{caller}: output names must be non-empty strings")
    padded = [name for name in output_names if name != name.strip()]
    if padded:
        raise ValueError(
            f"{caller}: output names must not contain surrounding "
            f"whitespace: {padded}"
        )
    if len(set(output_names)) != len(output_names):
        raise ValueError(f"{caller}: output names must be unique")

    collisions = [name for name in output_names if name in xds.variables]
    if collisions:
        raise ValueError(
            f"{caller}: output variables already exist: {collisions}"
        )
    unsafe = [name for name in output_names if _NUMERIC_SUFFIX.fullmatch(name)]
    if unsafe:
        raise ValueError(
            f"{caller}: output names must not end with an underscore "
            f"followed by digits because Anemoi treats that as a level suffix: {unsafe}"
        )
    return output_names


def ocean_density(
    xds: xr.Dataset,
    temperature: str = "temp",
    salinity: str = "so",
    name: str = "rho",
    ref_pressure: float | None = 0.0,
    subtract_1000: bool = False,
    eos: str | None = "wright",
) -> xr.Dataset:
    """Append Wright (1997) seawater density on the native ocean levels.

    A finite ``ref_pressure`` produces potential density referenced to that
    pressure. ``None`` produces approximate in-situ density using the
    Boussinesq pressure ``RHO_0 * GRAVITY * level``. Density is diagnosed before
    finite-volume regridding and remains a three-dimensional field.
    """
    caller = "ocean_density"
    temp, salt = _validate_inputs(xds, temperature, salinity, caller)
    level = _validate_level(xds, caller)
    output_name = _validate_output_names(
        [name],
        expected_count=1,
        xds=xds,
        caller=caller,
        value_description="density fields",
    )[0]
    resolved_eos = _resolve_eos(eos, caller)
    if not isinstance(subtract_1000, (bool, np.bool_)):
        raise TypeError(f"{caller}: subtract_1000 must be a boolean")

    if ref_pressure is None:
        pressure = xr.DataArray(
            hydrostatic_pressure(level),
            coords={"level": xds["level"]},
            dims=("level",),
        )
        density_kind = "in_situ"
        pressure_description = (
            f"hydrostatic p = {RHO_0:g} kg m-3 * {GRAVITY:g} m s-2 * level"
        )
    else:
        if isinstance(ref_pressure, (bool, np.bool_)):
            raise TypeError(
                f"{caller}: ref_pressure must be a real number or null"
            )
        try:
            pressure = float(ref_pressure)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{caller}: ref_pressure must be a real number or null"
            ) from error
        if not np.isfinite(pressure) or pressure < 0.0:
            raise ValueError(
                f"{caller}: ref_pressure must be finite and nonnegative"
            )
        density_kind = "potential"
        pressure_description = f"constant reference pressure {pressure:g} Pa"

    density = xr.apply_ufunc(
        wright_density,
        temp,
        salt,
        pressure,
        kwargs={"eos": resolved_eos},
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    if subtract_1000:
        density = density - 1000.0

    density = density.rename(output_name)
    density.attrs = {
        "diagnostic": "ocean_density",
        "long_name": (
            "sea water sigma" if subtract_1000 else "sea water density"
        ),
        "units": "kg m-3",
        "description": (
            f"{density_kind} seawater density from the {resolved_eos} equation "
            f"of state using {pressure_description}"
            + (", minus 1000 kg m-3" if subtract_1000 else "")
        ),
        "density_kind": density_kind,
        "equation_of_state": resolved_eos,
        "subtract_1000": bool(subtract_1000),
        "sigma_offset_kg_m3": 1000.0 if subtract_1000 else 0.0,
        "source_temperature": temperature,
        "source_salinity": salinity,
        "vertical_coordinate": "nominal_layer_center_depth",
        "source_vertical_level_count": int(level.size),
        "land_policy": "NaN where temperature or salinity is invalid",
        "missing_policy": "elementwise finite temperature and salinity",
        "provenance": (
            "Wright (1997), ported from MOM6 MOM_EOS_Wright_full.F90 / "
            "MOM_EOS_Wright_red.F90"
        ),
    }
    if ref_pressure is None:
        density.attrs.update(
            {
                "pressure_approximation": "p = rho0 * gravity * nominal depth",
                "reference_density_kg_m3": float(RHO_0),
                "gravity_m_s2": float(GRAVITY),
            }
        )
    else:
        density.attrs["reference_pressure_pa"] = float(pressure)

    xds[output_name] = density
    return xds


def mixed_layer_depth(
    xds: xr.Dataset,
    temperature: str = "temp",
    salinity: str = "so",
    thresholds: Sequence[float] | np.ndarray | float = (0.03,),
    names: Sequence[str] | str | None = None,
    ref_pressure: float = 0.0,
    eos: str | None = "wright",
    pathological_to_nan: bool = True,
    mld_search_depth: float = 200.0,
    min_levels_in_search: int = 10,
    max_level_spacing: float = 25.0,
    expected_level_count: int | None = 75,
    minimum_deepest_level: float | None = 5_000.0,
) -> xr.Dataset:
    """Append MOM6 density-threshold mixed layer depth fields.

    Temperature must be potential temperature in degrees Celsius, salinity must
    be practical salinity in psu, and ``level`` must contain native positive-down
    layer-center depths in metres. Every requested threshold is calculated by a
    single memory-bounded vertical pass before any later vertical regridding.
    """
    temp, salt = _validate_inputs(
        xds,
        temperature,
        salinity,
        "mixed_layer_depth",
    )
    level = _validate_level(xds, "mixed_layer_depth", minimum_levels=2)
    grid_description = check_vertical_resolution(
        level,
        mld_search_depth=mld_search_depth,
        min_levels_in_search=min_levels_in_search,
        max_level_spacing=max_level_spacing,
        caller="mixed_layer_depth",
        expected_level_count=expected_level_count,
        minimum_deepest_level=minimum_deepest_level,
    )
    _validate_level_chunks("mixed_layer_depth", temp, salt)

    density_thresholds = _as_thresholds(thresholds)
    output_names = _resolve_names(names, density_thresholds.size, xds)
    resolved_eos = _resolve_eos(eos, "mixed_layer_depth")
    try:
        reference_pressure = float(ref_pressure)
    except (TypeError, ValueError) as error:
        raise NotImplementedError(
            "mixed_layer_depth: only ref_pressure=0 is implemented"
        ) from error
    if not np.isfinite(reference_pressure) or reference_pressure != 0.0:
        raise NotImplementedError(
            "mixed_layer_depth: only ref_pressure=0 is implemented; "
            f"got {ref_pressure!r}"
        )
    if not isinstance(pathological_to_nan, (bool, np.bool_)):
        raise TypeError("mixed_layer_depth: pathological_to_nan must be a boolean")

    logger.info(
        "mixed_layer_depth: using %d levels spanning %.4g-%.4g m for thresholds %s",
        level.size,
        level[0],
        level[-1],
        density_thresholds.tolist(),
    )

    threshold_dim = "_mld_threshold"
    diagnosed = xr.apply_ufunc(
        _mixed_layer_depth_for_thresholds,
        temp,
        salt,
        input_core_dims=[["level"], ["level"]],
        output_core_dims=[[threshold_dim]],
        kwargs={
            "depth": level,
            "thresholds": density_thresholds,
            "ref_pressure": reference_pressure,
            "eos": resolved_eos,
            "pathological_to_nan": pathological_to_nan,
        },
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            "output_sizes": {threshold_dim: density_thresholds.size},
            "allow_rechunk": False,
        },
    )
    diagnosed = diagnosed.assign_coords({threshold_dim: density_thresholds})

    for index, (name, threshold) in enumerate(
        zip(output_names, density_thresholds, strict=True)
    ):
        output = diagnosed.isel({threshold_dim: index}, drop=True).rename(name)
        output.attrs = {
            "diagnostic": "mixed_layer_depth",
            "long_name": "ocean mixed layer depth",
            "units": "m",
            "description": (
                "first depth where potential density exceeds the top-layer "
                f"value by {threshold:g} kg m-3"
            ),
            "density_threshold": float(threshold),
            "density_threshold_units": "kg m-3",
            "reference_pressure": reference_pressure,
            "reference_pressure_pa": reference_pressure,
            "reference_pressure_units": "Pa",
            "reference_density_layer": "top_model_layer",
            "equation_of_state": resolved_eos,
            "source_temperature": temperature,
            "source_salinity": salinity,
            "vertical_grid": grid_description,
            "vertical_coordinate": "nominal_layer_center_depth",
            "source_vertical_level_count": int(level.size),
            "land_policy": "NaN where the surface temperature or salinity is invalid",
            "bottom_policy": "deepest valid layer center when the threshold is not crossed",
            "missing_policy": (
                "joint finite temperature and salinity from the top model layer "
                "until the first invalid level"
            ),
            "pathological_policy": (
                "NaN" if pathological_to_nan else "MOM6-compatible zero"
            ),
            "provenance": (
                "MOM6 diagnoseMLDbyDensityDifference (MOM_diagnose_MLD.F90) "
                "with Wright (1997) potential density"
            ),
        }
        xds[name] = output

    return xds


def _as_ohc_depths(
    depths: Sequence[float | None] | np.ndarray | float | None,
) -> tuple[float | None, ...]:
    """Normalize configured integration depths for the OHC kernel."""
    if isinstance(depths, (str, bytes)):
        raise TypeError(
            "ocean_heat_content: depths must be numeric, null, or a sequence"
        )
    if depths is None:
        candidates = [None]
    elif isinstance(depths, np.ndarray):
        candidates = [depths.item()] if depths.ndim == 0 else depths
    elif isinstance(depths, Sequence):
        candidates = list(depths)
    else:
        candidates = [depths]
    return _validated_ohc_depths(candidates)


def _format_ohc_depth(depth: float) -> str:
    """Format a validated depth for an unambiguous Anemoi channel name."""
    value = float(depth)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def _resolve_ohc_names(
    names: Sequence[str] | str | None,
    depths: tuple[float | None, ...],
    xds: xr.Dataset,
) -> list[str]:
    """Resolve and validate one OHC channel name per integration depth."""
    if names is None:
        output_names = [
            "ohc" if depth is None else f"ohc{_format_ohc_depth(depth)}"
            for depth in depths
        ]
    elif isinstance(names, str):
        output_names = [names]
    elif not isinstance(names, Sequence):
        raise TypeError(
            "ocean_heat_content: names must be a string or sequence of strings"
        )
    else:
        output_names = list(names)
    return _validate_output_names(
        output_names,
        expected_count=len(depths),
        xds=xds,
        caller="ocean_heat_content",
        value_description="depths",
    )


def _resolve_ohc_output_units(output_units: str) -> tuple[str, float]:
    """Return canonical OHC units and the scale from J m-2."""
    if not isinstance(output_units, str):
        raise TypeError("ocean_heat_content: output_units must be a string")
    key = (
        output_units.strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("^", "")
        .replace("/", "")
    )
    try:
        return _OHC_OUTPUT_UNITS[key]
    except KeyError as error:
        raise ValueError(
            "ocean_heat_content: output_units must describe 'GJ m-2' or "
            f"'J m-2'; got {output_units!r}"
        ) from error


def ocean_heat_content(
    xds: xr.Dataset,
    temperature: str = "temp",
    depths: Sequence[float | None] | np.ndarray | float | None = (
        700.0,
        2000.0,
        None,
    ),
    names: Sequence[str] | str | None = None,
    interfaces: Sequence[float] | np.ndarray | None = None,
    rho0: float = RHO_0,
    heat_capacity: float = HEAT_CAPACITY,
    require_full_depth: bool = True,
    output_units: str = "GJ m-2",
) -> xr.Dataset:
    """Append finite-volume ocean heat content for one or more depths.

    The memory-bounded kernel computes ``rho0 * Cp * integral(temp dz)`` in
    J m-2 over the native, contiguous water column. This wrapper scales the
    result to GJ m-2 by default before Anemoi can cast it for storage. Set
    ``output_units`` to ``"J m-2"`` only when the larger magnitude is desired.
    """
    caller = "ocean_heat_content"
    temp = _validate_temperature(xds, temperature, caller)
    level = _validate_level(xds, caller)
    _validate_level_chunks(caller, temp)
    integration_depths = _as_ohc_depths(depths)
    output_names = _resolve_ohc_names(names, integration_depths, xds)
    canonical_units, scale_from_joules = _resolve_ohc_output_units(output_units)
    density = _validated_positive_constant("ocean_heat_content: rho0", rho0)
    capacity = _validated_positive_constant(
        "ocean_heat_content: heat_capacity",
        heat_capacity,
    )
    if not isinstance(require_full_depth, (bool, np.bool_)):
        raise TypeError(f"{caller}: require_full_depth must be a boolean")
    require_full_depth = bool(require_full_depth)

    if interfaces is None:
        interface_values = interfaces_from_centers(level)
        interface_source = "reconstructed_from_level_centers"
        vertical_coordinate = (
            "nominal_layer_interfaces_reconstructed_from_level"
        )
    else:
        interface_values = _validated_interfaces(interfaces, level.size)
        if np.any(level <= interface_values[:-1]) or np.any(
            level >= interface_values[1:]
        ):
            raise ValueError(
                f"{caller}: every level center must lie strictly within its explicit "
                "interface pair"
            )
        interface_source = "explicit"
        vertical_coordinate = "explicit_nominal_layer_interfaces"

    if canonical_units == "J m-2":
        logger.warning(
            "%s: J m-2 values are commonly 1e10-1e11 and are unsafe for "
            "downstream fp16; prefer output_units='GJ m-2'",
            caller,
        )
    for name, depth in zip(output_names, integration_depths, strict=True):
        if depth is not None and depth > interface_values[-1]:
            logger.warning(
                "%s: requested depth %.6g m is below the deepest interface "
                "%.6g m; %r will be all NaN when require_full_depth is true",
                caller,
                depth,
                interface_values[-1],
                name,
            )

    logger.info(
        "%s: using %d levels spanning %.4g-%.4g m for depths %s",
        caller,
        level.size,
        level[0],
        level[-1],
        list(integration_depths),
    )
    depth_dim = "_ohc_depth"
    diagnosed_joules = xr.apply_ufunc(
        _ocean_heat_content_for_depths,
        temp,
        input_core_dims=[["level"]],
        output_core_dims=[[depth_dim]],
        kwargs={
            "interfaces": interface_values,
            "max_depths": integration_depths,
            "rho0": density,
            "heat_capacity": capacity,
            "require_full_depth": require_full_depth,
        },
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            "output_sizes": {depth_dim: len(integration_depths)},
            "allow_rechunk": False,
        },
    )
    diagnosed = diagnosed_joules * scale_from_joules

    for index, (name, depth) in enumerate(
        zip(output_names, integration_depths, strict=True)
    ):
        output = diagnosed.isel({depth_dim: index}, drop=True).rename(name)
        span = "the full contiguous water column"
        if depth is not None:
            span = f"0-{depth:g} m"
        if depth is None:
            bottom_policy = "integrate to the deepest contiguous valid interface"
        elif require_full_depth:
            bottom_policy = (
                "NaN unless the contiguous water column reaches the requested depth"
            )
        else:
            bottom_policy = (
                "integrate to the shallower of the requested depth and local bottom"
            )
        output.attrs = {
            "diagnostic": "ocean_heat_content",
            "long_name": "ocean heat content",
            "units": canonical_units,
            "description": (
                f"rho0*Cp*integral(temp dz) over {span}, relative to 0 degC"
            ),
            "native_units": "J m-2",
            "joules_to_output_units": float(scale_from_joules),
            "source_temperature": temperature,
            "temperature_reference": 0.0,
            "temperature_reference_units": "degC",
            "rho0": density,
            "rho0_units": "kg m-3",
            "heat_capacity": capacity,
            "heat_capacity_units": "J kg-1 K-1",
            "require_full_depth": require_full_depth,
            "interface_source": interface_source,
            "layer_thickness_source": "nominal interfaces",
            "time_varying_layer_thickness_used": False,
            "vertical_coordinate": vertical_coordinate,
            "source_vertical_level_count": int(level.size),
            "land_policy": "NaN where surface temperature is invalid",
            "bottom_policy": bottom_policy,
            "missing_policy": (
                "contiguous finite temperature from the surface until the first "
                "invalid level"
            ),
            "provenance": (
                "Boussinesq finite-volume heat content following MOM6 RHO_0 "
                "and C_P conventions"
            ),
        }
        if depth is not None:
            output.attrs.update(
                {
                    "integration_depth": float(depth),
                    "integration_depth_units": "m",
                }
            )
        xds[name] = output

    return xds
