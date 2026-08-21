"""Dataset level wrappers for the ocean diagnostics.

These are the functions the ``transforms`` section of a recipe reaches, one per
recognized key: ``ocean_density``, ``mixed_layer_depth`` and
``ocean_heat_content``. The physics lives in :mod:`ufs2arco.ocean_diagnostics`;
this module handles variable lookup, the vertical grid, naming, metadata, and
dask.

All three run before any vertical regridding, because
:meth:`ufs2arco.transforms.Transformer.__call__` applies operations in a fixed
order rather than the order they appear in the yaml. That matters most for
mixed layer depth: the mixed layer lives in the top tens of meters, so computing
it on a handful of coarsened layers rather than the native column would smear
the pycnocline and flatten out the field's spatial and seasonal structure.
"""

import logging
from typing import Optional

import numpy as np
import xarray as xr

from ufs2arco.ocean_diagnostics import (
    DEFAULT_EOS,
    GRAVITY,
    HEAT_CAPACITY,
    RHO_0,
    available_eos,
    hydrostatic_pressure,
    interfaces_from_centers,
    mixed_layer_depth_by_density_difference,
    ocean_heat_content as _ocean_heat_content_kernel,
    wright_density,
)

logger = logging.getLogger("ufs2arco")


def _resolve_eos(eos: str) -> str:
    """Map a user facing equation of state name onto an implemented one.

    ``"wright"`` is accepted as an alias for MOM6's own default, which is
    ``WRIGHT_FULL`` (``MOM_EOS.F90``, ``EOS_DEFAULT``).
    """
    if eos is None:
        return DEFAULT_EOS
    key = str(eos).strip().lower()
    if key == "wright":
        return DEFAULT_EOS
    if key not in available_eos():
        raise NotImplementedError(
            f"unrecognized equation of state {eos!r}; implemented options are "
            f"{available_eos() + ('wright',)}, where 'wright' aliases {DEFAULT_EOS!r}"
        )
    return key


def _require(xds: xr.Dataset, *names: str, caller: str) -> None:
    """Raise a useful error when an expected input variable is absent."""
    missing = [name for name in names if name not in xds]
    if missing:
        raise KeyError(
            f"{caller}: can't find {missing} in dataset, needed to compute this diagnostic. "
            f"Available data variables are {sorted(xds.data_vars)}"
        )
    if "level" not in xds.dims:
        raise KeyError(
            f"{caller}: dataset has no 'level' dimension, so there is no vertical column to "
            f"work with. Dimensions are {sorted(xds.dims)}"
        )


def _levels(xds: xr.Dataset, caller: str) -> np.ndarray:
    """Return the vertical coordinate as increasing depths, or explain why not."""
    if "level" not in xds.coords:
        raise KeyError(
            f"{caller}: dataset has a 'level' dimension but no 'level' coordinate, so layer "
            "depths are unknown"
        )
    level = np.asarray(xds["level"].values, dtype=float)
    if level.ndim != 1:
        raise ValueError(f"{caller}: expected a 1D 'level' coordinate, got shape {level.shape}")
    if not np.all(np.diff(level) > 0):
        raise ValueError(
            f"{caller}: 'level' must increase monotonically with depth. Got a coordinate "
            f"running from {level[0]:g} to {level[-1]:g}"
        )
    return level


def _log_grid(caller: str, level: np.ndarray) -> None:
    """Record the vertical grid actually used, so runs are auditable from the log."""
    logger.info(
        f"{caller}: using {level.size} vertical levels spanning "
        f"{level[0]:.4g} - {level[-1]:.4g} m"
    )


def check_vertical_resolution(
    level: np.ndarray,
    mld_search_depth: float = 200.0,
    min_levels_in_search: int = 10,
    max_level_spacing: float = 25.0,
    caller: str = "check_vertical_resolution",
) -> str:
    """Refuse a vertical grid too coarse to resolve a mixed layer.

    Mixed layer depth can never be resolved more finely than the layer spacing.
    On a grid with, say, interfaces at 0, 5, 20, 60, 100 and 200 m, a pycnocline
    anywhere between 60 and 100 m is averaged over a 40 m layer, the surface to
    layer density difference is diluted, and the threshold ends up crossed in the
    same layer nearly everywhere. The result is a smooth looking field with the
    wrong values and no seasonal cycle.

    Checking only that the surface is present does not catch this, since that
    grid's topmost center is 2.5 m. The test has to be on spacing.

    :meth:`ufs2arco.transforms.Transformer.__call__` already makes it impossible
    to express "regrid, then compute mixed layer depth" in a recipe. This guard
    covers the paths ordering cannot reach: ``source.levels`` and
    ``source.slices.sel.level`` subset the column in ``Source.__init__``, before
    a ``Transformer`` exists, and ``source.uri`` may point at an already
    coarsened store.

    For calibration, the replay native grid has 30 levels above 200 m with a
    maximum spacing of 17.44 m, and a topmost center at 0.52 m, so it clears the
    defaults with room to spare. Note that the same grid coarsens to 26.55 m
    spacing by 300 m, which is why the search depth stops at 200 m rather than
    reaching deeper.

    Args:
        level (numpy.ndarray): layer center depths, increasing [m]
        mld_search_depth (float, optional): the range a mixed layer is expected
            to live in [m]
        min_levels_in_search (int, optional): minimum levels required above
            ``mld_search_depth``
        max_level_spacing (float, optional): maximum tolerated spacing between
            consecutive levels within that range [m]
        caller (str, optional): name used in error messages

    Returns:
        str: a description of the grid that passed, for use as a variable attr

    Raises:
        ValueError: if the grid is too coarse or too shallow
    """
    hint = (
        "This usually means the column was subset before any transform ran, via "
        "'source.levels' or 'source.slices.sel.level', or that 'source.uri' points at an "
        "already vertically coarsened store. Ocean diagnostics need the native column."
    )

    search = level[level < mld_search_depth]
    if search.size < min_levels_in_search:
        raise ValueError(
            f"{caller}: found only {search.size} vertical level(s) above {mld_search_depth:g} m, "
            f"but at least {min_levels_in_search} are needed to resolve a mixed layer. "
            f"The full coordinate spans {level[0]:.4g} - {level[-1]:.4g} m across {level.size} "
            f"level(s). {hint}"
        )

    spacing = np.diff(search)
    worst = float(spacing.max())
    if worst > max_level_spacing:
        index = int(np.argmax(spacing))
        raise ValueError(
            f"{caller}: vertical spacing reaches {worst:.4g} m (between {search[index]:.4g} m "
            f"and {search[index + 1]:.4g} m), which exceeds the {max_level_spacing:g} m limit "
            f"for resolving a mixed layer. Mixed layer depth cannot be resolved more finely "
            f"than the layer spacing. {hint}"
        )

    return (
        f"validated vertical grid: {search.size} levels above {mld_search_depth:g} m, "
        f"spacing {float(spacing.min()):.4g} - {worst:.4g} m, topmost center {level[0]:.4g} m"
    )


def ocean_density(
    xds: xr.Dataset,
    temperature: Optional[str] = "temp",
    salinity: Optional[str] = "so",
    name: Optional[str] = "rho",
    ref_pressure: Optional[float] = 0.0,
    subtract_1000: Optional[bool] = False,
    eos: Optional[str] = "wright",
) -> xr.Dataset:
    """Add seawater density as a 3D field on the existing vertical levels.

    Uses the Wright (1997) equation of state ported from MOM6, see
    :func:`ufs2arco.ocean_diagnostics.wright_density`. This is elementwise, so it
    stays lazy under dask without any special handling.

    With ``ref_pressure=0`` the result is potential density referenced to the
    surface, the same quantity mixed layer depth is defined on. With
    ``ref_pressure=None`` it is in situ density, using a Boussinesq
    ``p = rho0*g*z`` built from the ``level`` coordinate.

    Because this keeps the ``level`` dimension, the new variable flows through the
    rest of the pipeline exactly like temperature: a later
    ``fv_vertical_regrid_ocean`` thickness averages and bottom masks it, and the
    anemoi target expands it into ``rho_5``, ``rho_100`` and so on.

    Args:
        xds (xr.Dataset): with ``temperature`` and ``salinity`` on a ``level`` dim
        temperature (str, optional): potential temperature variable [degC]
        salinity (str, optional): practical salinity variable [psu]
        name (str, optional): name for the new variable
        ref_pressure (float, optional): reference pressure [Pa], or None for in situ
        subtract_1000 (bool, optional): emit sigma, i.e. density minus 1000 kg m-3
        eos (str, optional): equation of state, see :func:`_resolve_eos`

    Returns:
        xds (xr.Dataset): with the density variable added
    """
    caller = "ocean_density"
    _require(xds, temperature, salinity, caller=caller)
    if name in xds:
        raise ValueError(
            f"{caller}: {name!r} is already in the dataset; choose a different 'name'"
        )
    eos = _resolve_eos(eos)
    level = _levels(xds, caller)
    _log_grid(caller, level)

    if ref_pressure is None:
        pressure = xr.DataArray(
            hydrostatic_pressure(level),
            coords={"level": xds["level"]},
            dims="level",
        )
        reference = f"in situ, hydrostatic p = {RHO_0:g}*{GRAVITY:g}*z"
    else:
        pressure = float(ref_pressure)
        reference = f"potential density referenced to {pressure:g} Pa"

    rho = wright_density(xds[temperature], xds[salinity], pressure, eos=eos)
    if subtract_1000:
        rho = rho - 1000.0

    rho.attrs = {
        "long_name": "sea water sigma" if subtract_1000 else "sea water density",
        "units": "kg m-3",
        "description": (
            f"seawater density from the {eos} equation of state ({reference})"
            + (", minus 1000 kg m-3" if subtract_1000 else "")
            + f", computed from {temperature} and {salinity}"
        ),
        "equation_of_state": eos,
        "provenance": "Wright (1997), ported from MOM6 MOM_EOS_Wright_full.F90 / _red.F90",
    }
    xds[name] = rho
    return xds


def mixed_layer_depth(
    xds: xr.Dataset,
    temperature: Optional[str] = "temp",
    salinity: Optional[str] = "so",
    thresholds: Optional[list | tuple | float] = (0.03,),
    names: Optional[list | tuple | str] = None,
    ref_pressure: Optional[float] = 0.0,
    eos: Optional[str] = "wright",
    pathological_to_nan: Optional[bool] = True,
    mld_search_depth: Optional[float] = 200.0,
    min_levels_in_search: Optional[int] = 10,
    max_level_spacing: Optional[float] = 25.0,
) -> xr.Dataset:
    """Add mixed layer depth, from MOM6's density difference criterion.

    See :func:`ufs2arco.ocean_diagnostics.mixed_layer_depth_by_density_difference`
    for the algorithm and its two deliberate deviations from MOM6.

    The vertical grid is validated before any data is touched, so a recipe that
    would produce a degenerate field fails immediately with a specific message
    rather than writing something that looks plausible. See
    :func:`check_vertical_resolution`.

    Output names deliberately avoid a trailing ``_<digits>``: the anemoi target
    reads that pattern as a vertical level, so MOM6's own ``MLD_003`` would be
    silently reinterpreted as "MLD at level 3". The default for a single
    threshold is ``mld``; with several thresholds, pass ``names`` explicitly.

    Args:
        xds (xr.Dataset): with ``temperature`` and ``salinity`` on a ``level`` dim
        temperature (str, optional): potential temperature variable [degC]
        salinity (str, optional): practical salinity variable [psu]
        thresholds (float or sequence, optional): density thresholds [kg m-3].
            MOM6 uses 0.03 for ``MLD_003`` and 0.125 for ``MLD_0125``.
        names (str or sequence, optional): output names, one per threshold
        ref_pressure (float, optional): reference pressure [Pa]; only 0 is implemented
        eos (str, optional): equation of state, see :func:`_resolve_eos`
        pathological_to_nan (bool, optional): map MOM6's residual zeros to NaN
        mld_search_depth (float, optional): guard, see :func:`check_vertical_resolution`
        min_levels_in_search (int, optional): guard, as above
        max_level_spacing (float, optional): guard, as above

    Returns:
        xds (xr.Dataset): with one mixed layer depth variable per threshold
    """
    caller = "mixed_layer_depth"
    _require(xds, temperature, salinity, caller=caller)
    eos = _resolve_eos(eos)
    level = _levels(xds, caller)
    _log_grid(caller, level)

    grid_description = check_vertical_resolution(
        level,
        mld_search_depth=mld_search_depth,
        min_levels_in_search=min_levels_in_search,
        max_level_spacing=max_level_spacing,
        caller=caller,
    )

    thresholds = _as_sequence(thresholds)
    names = _resolve_names(names, thresholds, default_single="mld", caller=caller, kind="threshold")

    for name, threshold in zip(names, thresholds):
        if name in xds:
            raise ValueError(
                f"{caller}: {name!r} is already in the dataset; choose a different name"
            )
        result = xr.apply_ufunc(
            mixed_layer_depth_by_density_difference,
            xds[temperature],
            xds[salinity],
            kwargs={
                "depth": level,
                "density_diff": float(threshold),
                "ref_pressure": ref_pressure,
                "eos": eos,
                "pathological_to_nan": pathological_to_nan,
            },
            input_core_dims=[["level"], ["level"]],
            output_core_dims=[[]],
            dask="parallelized",
            output_dtypes=[float],
        )
        result.attrs = {
            "long_name": "ocean mixed layer depth",
            "units": "m",
            "description": (
                f"depth where potential density exceeds the near surface value by "
                f"{float(threshold):g} kg m-3, using the {eos} equation of state"
            ),
            "density_threshold": float(threshold),
            "equation_of_state": eos,
            "vertical_grid": grid_description,
            "provenance": (
                "MOM6 diagnoseMLDbyDensityDifference, MOM_diagnose_MLD.F90"
            ),
        }
        xds[name] = result

    return xds


def ocean_heat_content(
    xds: xr.Dataset,
    temperature: Optional[str] = "temp",
    depths: Optional[list | tuple | float] = (700, 2000, None),
    names: Optional[list | tuple | str] = None,
    interfaces: Optional[list | tuple] = None,
    rho0: Optional[float] = RHO_0,
    heat_capacity: Optional[float] = HEAT_CAPACITY,
    require_full_depth: Optional[bool] = True,
) -> xr.Dataset:
    """Add depth integrated ocean heat content.

    See :func:`ufs2arco.ocean_diagnostics.ocean_heat_content` for the integral and
    its partial cell weighting.

    Layer interfaces default to being reconstructed from the ``level``
    coordinate via
    :func:`ufs2arco.ocean_diagnostics.interfaces_from_centers`, so this works for
    any z-coordinate ocean source without a hard coded depth table. Pass
    ``interfaces`` to override.

    Default names are ``ohc300``, ``ohc1000`` and, for the full column, ``ohc``.
    As with mixed layer depth these avoid a trailing ``_<digits>``, which the
    anemoi target would read as a vertical level.

    Args:
        xds (xr.Dataset): with ``temperature`` on a ``level`` dim
        temperature (str, optional): potential temperature variable [degC]
        depths (float or sequence, optional): integration depths [m]; None means
            the full water column
        names (str or sequence, optional): output names, one per depth
        interfaces (sequence, optional): explicit interface depths [m],
            ``len(level) + 1`` of them
        rho0 (float, optional): Boussinesq reference density [kg m-3]
        heat_capacity (float, optional): heat capacity of seawater [J kg-1 K-1]
        require_full_depth (bool, optional): NaN where the column is shallower
            than the requested depth

    Returns:
        xds (xr.Dataset): with one heat content variable per depth
    """
    caller = "ocean_heat_content"
    _require(xds, temperature, caller=caller)
    level = _levels(xds, caller)
    _log_grid(caller, level)

    if interfaces is None:
        z_i = interfaces_from_centers(level)
    else:
        z_i = np.asarray(interfaces, dtype=float)
        if z_i.size != level.size + 1:
            raise ValueError(
                f"{caller}: got {z_i.size} interfaces for {level.size} levels, "
                f"expected {level.size + 1}"
            )

    depths = _as_sequence(depths)
    names = _resolve_names(names, depths, default_single=None, caller=caller, kind="depth")

    for name, max_depth in zip(names, depths):
        if name in xds:
            raise ValueError(
                f"{caller}: {name!r} is already in the dataset; choose a different name"
            )
        if max_depth is not None and float(max_depth) > z_i[-1]:
            logger.warning(
                f"{caller}: requested depth {float(max_depth):g} m is below the deepest "
                f"interface at {z_i[-1]:.4g} m, so {name!r} will be NaN everywhere when "
                "require_full_depth is True"
            )
        result = xr.apply_ufunc(
            _ocean_heat_content_kernel,
            xds[temperature],
            kwargs={
                "interfaces": z_i,
                "max_depth": None if max_depth is None else float(max_depth),
                "rho0": rho0,
                "heat_capacity": heat_capacity,
                "require_full_depth": require_full_depth,
            },
            input_core_dims=[["level"]],
            output_core_dims=[[]],
            dask="parallelized",
            output_dtypes=[float],
        )
        span = "the full water column" if max_depth is None else f"0 - {float(max_depth):g} m"
        result.attrs = {
            "long_name": "ocean heat content",
            "units": "J m-2",
            "description": (
                f"heat content integrated over {span} as rho0*Cp*integral(theta dz), "
                f"with rho0={rho0:g} kg m-3 and Cp={heat_capacity:g} J kg-1 K-1"
            ),
            "rho0": float(rho0),
            "heat_capacity": float(heat_capacity),
            "require_full_depth": str(bool(require_full_depth)),
            "provenance": "Boussinesq heat content following MOM6 RHO_0 and C_P conventions",
        }
        if max_depth is not None:
            result.attrs["integration_depth"] = float(max_depth)
        xds[name] = result

    return xds


def _as_sequence(value) -> list:
    """Accept a scalar, None, or a sequence, and always return a list."""
    if value is None:
        return [None]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_names(names, values, default_single, caller, kind) -> list:
    """Work out output variable names, or explain what the recipe must provide."""
    if names is not None:
        names = [names] if isinstance(names, str) else list(names)
        if len(names) != len(values):
            raise ValueError(
                f"{caller}: got {len(names)} name(s) for {len(values)} {kind}(s); "
                "provide exactly one name per value"
            )
        return names

    if kind == "depth":
        return ["ohc" if value is None else f"ohc{_format_depth(value)}" for value in values]

    if len(values) == 1 and default_single is not None:
        return [default_single]

    raise ValueError(
        f"{caller}: with {len(values)} {kind}s you must supply 'names', one per {kind}. "
        f"Only a single {kind} gets a default name ({default_single!r}). Note that names "
        "ending in an underscore followed by digits are read as vertical levels by the "
        "anemoi target, so avoid e.g. 'mld_003'."
    )


def _format_depth(value) -> str:
    """Render a depth for use in a variable name, without a decimal point."""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")
