import logging
import os
from functools import lru_cache
from typing import Optional

import numpy as np
import xarray as xr
import yaml

from ufs2arco.ocean_diagnostics import interfaces_from_centers

try:
    import flox
    _has_flox = True
except ImportError:
    _has_flox = False

logger = logging.getLogger("ufs2arco")


def _validated_bool(value: bool, *, name: str) -> bool:
    """Return a strict Boolean configuration value."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _validated_interfaces(
    values: list | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return finite, strictly increasing one-dimensional interfaces."""
    try:
        raw_interfaces = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain real numeric values") from error

    if (
        not np.issubdtype(raw_interfaces.dtype, np.number)
        or np.issubdtype(raw_interfaces.dtype, np.bool_)
        or np.issubdtype(raw_interfaces.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must contain real numeric values")
    interfaces = np.asarray(raw_interfaces, dtype=np.float64)

    if interfaces.ndim != 1 or interfaces.size < 2:
        raise ValueError(f"{name} must contain at least two 1D values")
    if not np.all(np.isfinite(interfaces)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all(np.diff(interfaces) > 0.0):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return interfaces


def _validated_source_interfaces(
    values: list | np.ndarray,
    levels: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Validate source-cell interfaces against their center coordinate."""
    interfaces = _validated_interfaces(values, name=name)
    if interfaces.size != levels.size + 1:
        raise ValueError(
            f"{name} has {interfaces.size} values for {levels.size} source "
            f"levels; expected {levels.size + 1}"
        )
    if interfaces[0] != 0.0:
        raise ValueError(f"{name} must start at the ocean surface (0 m)")
    if np.any(levels <= interfaces[:-1]) or np.any(levels >= interfaces[1:]):
        raise ValueError(
            f"{name} must bracket every source level center strictly"
        )
    return interfaces


@lru_cache(maxsize=1)
def _packaged_replay_ocean_interfaces() -> np.ndarray:
    """Load the native Replay MOM6 interfaces shipped with the package."""
    cfg_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "replay_ocean_vertical_levels.yaml",
        )
    )
    with open(cfg_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    interfaces = np.asarray(config["z_i"], dtype=np.float64)
    interfaces.setflags(write=False)
    return interfaces


def _resolve_ocean_source_interfaces(
    xds: xr.Dataset,
    source_interfaces: Optional[list | np.ndarray],
) -> tuple[np.ndarray, str]:
    """Resolve native interfaces while allowing non-Replay z coordinates."""
    if "level" not in xds.coords:
        raise KeyError(
            "fv_vertical_regrid_ocean: expected a 'level' coordinate"
        )
    levels = np.asarray(xds["level"].values, dtype=np.float64)
    if (
        levels.ndim != 1
        or not np.all(np.isfinite(levels))
        or not np.all(np.diff(levels) > 0.0)
    ):
        raise ValueError(
            "fv_vertical_regrid_ocean: 'level' must be finite, 1D, and "
            "strictly increase from the surface downward"
        )

    if source_interfaces is not None:
        return (
            _validated_source_interfaces(
                source_interfaces,
                levels,
                name="source_interfaces",
            ),
            "configured source_interfaces",
        )

    for coordinate_name in ("interface", "z_i"):
        if coordinate_name in xds:
            return (
                _validated_source_interfaces(
                    xds[coordinate_name].values,
                    levels,
                    name=f"dataset coordinate {coordinate_name!r}",
                ),
                f"dataset coordinate {coordinate_name!r}",
            )

    replay_interfaces = _packaged_replay_ocean_interfaces()
    replay_centers = 0.5 * (
        replay_interfaces[:-1] + replay_interfaces[1:]
    )
    if levels.size == replay_centers.size and np.allclose(
        levels,
        replay_centers,
        rtol=0.0,
        atol=5.0e-6,
    ):
        return (
            _validated_source_interfaces(
                replay_interfaces,
                levels,
                name="packaged Replay z_i",
            ),
            "packaged Replay z_i",
        )

    replay_center_matches = np.isclose(
        levels[:, np.newaxis],
        replay_centers[np.newaxis, :],
        rtol=0.0,
        atol=5.0e-6,
    )
    if levels.size < replay_centers.size and np.all(
        replay_center_matches.any(axis=1)
    ):
        raise ValueError(
            "fv_vertical_regrid_ocean: detected a subset of the native "
            "Replay ocean levels; use the full native column or provide "
            "explicit valid source_interfaces"
        )

    reconstructed = interfaces_from_centers(levels)
    return (
        _validated_source_interfaces(
            reconstructed,
            levels,
            name="interfaces reconstructed from level centers",
        ),
        "reconstructed from level centers",
    )


def _fv_vertical_regrid(
    xds: xr.Dataset,
    weight_var: str,
    interfaces: list | np.ndarray,
    use_nearest_interfaces: Optional[bool] = False,
    keep_weight_var: Optional[bool] = False,
    _valid_mask: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """Apply the shared vertical reduction with optional physical support.

    Args:
        xds (xr.Dataset)
        weight_var (str): with the name of the variable to weight regridding by
        interfaces (array_like): array of interface values to average vertical coordinate "level" between
            if "interface" does not exist in the dataset, this will be added, otherwise we subsample
        use_nearest_interfaces (bool, optional): if True, and "interface" is in the dataset,
            grab interfaces from dataset via
            ``xarray.Dataset.sel(interface=interfaces, method='nearest')``
        keep_weight_var (bool, optional): if False, drop the weight variable from the dataset before saving
        _valid_mask (xr.DataArray, optional): internal source-cell support mask

    Returns:
        xds (xr.Dataset): with vertical averaging
    """

    if not _has_flox:
        logger.warning("Could not import flox, install with 'conda install -c conda-forge flox' for faster volume averaging (i.e. groupby operations)")

    if weight_var not in xds:
        raise KeyError(
            f"fv_vertical_regrid: can't find {weight_var!r} in dataset, "
            "so it cannot be used for regridding"
        )

    use_nearest_interfaces = _validated_bool(
        use_nearest_interfaces,
        name="fv_vertical_regrid use_nearest_interfaces",
    )
    keep_weight_var = _validated_bool(
        keep_weight_var,
        name="fv_vertical_regrid keep_weight_var",
    )
    requested_interfaces = _validated_interfaces(
        interfaces,
        name="fv_vertical_regrid interfaces",
    )

    # get or create the exact interface values
    if "interface" in xds:
        kw = {"method": "nearest"} if use_nearest_interfaces else {}
        try:
            source_interfaces = _validated_interfaces(
                xds["interface"].values,
                name="fv_vertical_regrid source interface coordinate",
            )
            if use_nearest_interfaces and (
                requested_interfaces[0] < source_interfaces[0]
                or requested_interfaces[-1] > source_interfaces[-1]
            ):
                raise ValueError(
                    "requested interfaces must lie within the source interface "
                    "range when use_nearest_interfaces=True"
                )
            xds = xds.sel(interface=requested_interfaces, **kw)
            _validated_interfaces(
                xds["interface"].values,
                name="fv_vertical_regrid resolved interfaces",
            )
        except (KeyError, ValueError) as error:
            message = "fv_vertical_regrid: couldn't resolve specified interfaces"
            if not use_nearest_interfaces:
                message += (
                    "\nSet use_nearest_interfaces=True to snap requested "
                    "boundaries to the native interface coordinate."
                )
            raise ValueError(message) from error

    else:
        xds["interface"] = xr.DataArray(
            requested_interfaces,
            coords={"interface": requested_interfaces},
            dims="interface",
            attrs={"description": "vertical coordinate interfaces created for regridding"},
        )

    # get the regrid weighting
    finite_weight = xds[weight_var].where(np.isfinite(xds[weight_var]))
    layer_thickness = finite_weight.groupby_bins(
        "level",
        bins=xds["interface"],
    ).sum(skipna=True, min_count=1)
    layer_thickness.attrs = xds[weight_var].attrs.copy()

    # do the regridding
    vars3d = [x for x in xds.data_vars if "level" in xds[x].dims and x != weight_var]
    for key in vars3d:
        attrs = xds[key].attrs.copy()
        value = xds[key]
        if _valid_mask is not None:
            value = value.where(_valid_mask)
        broadcast_weight = finite_weight.broadcast_like(value)
        valid_weight = broadcast_weight.where(
            np.isfinite(value) & np.isfinite(broadcast_weight)
        )
        numerator = (value * valid_weight).groupby_bins(
            "level",
            bins=xds["interface"],
        ).sum(skipna=True, min_count=1)
        denominator = valid_weight.groupby_bins(
            "level",
            bins=xds["interface"],
        ).sum(skipna=True, min_count=1)
        xds[key] = numerator / denominator
        xds[key].attrs = attrs
        long_name = attrs.get("long_name", key)
        xds[key].attrs["long_name"] = f"vertically regridded {long_name}"

    # make new coordinates for approximate new level
    new_level = (xds["interface"].values[:-1] + xds["interface"].values[1:])/2
    if np.all(new_level == new_level.astype(int)):
        new_level = new_level.astype(int)
    xds["new_level"] = xr.DataArray(
        new_level,
        coords={"new_level": new_level},
        dims=("new_level",),
        attrs={
            "description": f"approximated vertical grid cell center after regridding",
            "details": "computed as (interface[:-1] + interface[1:])/2",
        },
    )
    # we need to drop the original weight_var (e.g. delz) b/c it has the OG levels on it
    # we'll add the regridded version later if desired
    xds = xds.drop_vars(weight_var)
    xds = xds.drop_vars("level")
    xds = xds.rename({"new_level": "level"})

    # handle the bins and add attrs
    xds["level_bins"] = xds["level_bins"].swap_dims({"level_bins": "level"})
    for key in vars3d:
        with xr.set_options(keep_attrs=True):
            xds[key] = xds[key].swap_dims({"level_bins": "level"})
        xds[key].attrs["vertical_coordinate"] = f"{weight_var} weighted average in vertical, new coordinate bounds represented by 'interface'"

    if keep_weight_var:
        xds[weight_var] = layer_thickness.swap_dims({"level_bins": "level"})
        xds[weight_var].attrs["vertical_coordinate"] = f"vertically averaged, new coordinate bounds represented by 'interface'"

    # unfortunately, cannot store the level_bins due to this issue: https://github.com/pydata/xarray/issues/2847
    xds = xds.drop_vars("level_bins")
    return xds


def fv_vertical_regrid(
    xds: xr.Dataset,
    weight_var: str,
    interfaces: list | np.ndarray,
    use_nearest_interfaces: Optional[bool] = False,
    keep_weight_var: Optional[bool] = False,
) -> xr.Dataset:
    """Vertically regrid a dataset based on interface values.

    Args:
        xds: Dataset to regrid.
        weight_var: Variable containing the source-cell weights.
        interfaces: Target vertical-coordinate interfaces.
        use_nearest_interfaces: Snap requests to an existing native interface
            coordinate when true.
        keep_weight_var: Retain the aggregated weight variable when true.

    Returns:
        Dataset with weighted vertical averages.
    """
    return _fv_vertical_regrid(
        xds,
        weight_var=weight_var,
        interfaces=interfaces,
        use_nearest_interfaces=use_nearest_interfaces,
        keep_weight_var=keep_weight_var,
    )


def fv_vertical_regrid_ocean(
    xds: xr.Dataset,
    interfaces: list | np.ndarray,
    weight_var: Optional[str] = "dz",
    use_nearest_interfaces: Optional[bool] = True,
    keep_weight_var: Optional[bool] = False,
    source_interfaces: Optional[list | np.ndarray] = None,
) -> xr.Dataset:
    """Vertically regrid ocean data and mask regridded layers below bathymetry.

    This resolves the native ocean interfaces, snaps requested target
    boundaries to them by default, and uses native layer thickness as the
    center-bin weight. The wet column is the leading contiguous set of finite
    temperature layers; target midpoints below its bottom interface are masked.

    ``source_interfaces`` overrides an interface coordinate on the input.
    Otherwise the wrapper uses an input ``interface``/``z_i`` coordinate, the
    packaged Replay interfaces when the full Replay center grid matches, or
    reconstructed midpoint interfaces for a different complete grid.
    """

    if "temp" not in xds:
        raise KeyError("fv_vertical_regrid_ocean: expected 'temp' in dataset to infer ocean bottom mask")

    use_nearest_interfaces = _validated_bool(
        use_nearest_interfaces,
        name="fv_vertical_regrid_ocean use_nearest_interfaces",
    )
    keep_weight_var = _validated_bool(
        keep_weight_var,
        name="fv_vertical_regrid_ocean keep_weight_var",
    )
    native_interfaces, interface_source = _resolve_ocean_source_interfaces(
        xds,
        source_interfaces,
    )

    xds = xds.copy()
    for coordinate_name in ("interface", "z_i"):
        interface_consumers = [
            name
            for name, variable in xds.data_vars.items()
            if coordinate_name in variable.dims and name != coordinate_name
        ]
        if interface_consumers:
            raise ValueError(
                "fv_vertical_regrid_ocean: source data variables must not use "
                f"the reserved {coordinate_name!r} dimension; found "
                f"{interface_consumers}"
            )
        if coordinate_name in xds:
            xds = xds.drop_vars(coordinate_name)

    finite_temperature = np.isfinite(xds["temp"])
    leading_wet = (
        finite_temperature.astype(np.int8)
        .cumprod(dim="level", dtype=np.int8)
        .astype(bool)
    )
    num_valid_layers = leading_wet.sum(dim="level").astype(int)
    index = num_valid_layers.clip(
        min=0,
        max=len(native_interfaces) - 1,
    ).astype(int)

    bottom_interface = xr.apply_ufunc(
        lambda indices: np.take(native_interfaces, indices),
        index,
        output_dtypes=[float],
        dask="parallelized" if index.chunks else None,
    )
    bottom_interface.name = "bottom_interface"

    dz = np.diff(native_interfaces)
    if len(dz) != len(xds["level"]):
        raise ValueError(
            "fv_vertical_regrid_ocean: computed layer thickness length "
            f"({len(dz)}) does not match source level length ({len(xds['level'])})"
        )

    xds[weight_var] = xr.DataArray(
        dz,
        coords={"level": xds["level"].values},
        dims="level",
        attrs={"long_name": "layer thickness"},
    )
    xds = xds.assign_coords(
        interface=xr.DataArray(
            native_interfaces,
            dims="interface",
            coords={"interface": native_interfaces},
        )
    )

    result = _fv_vertical_regrid(
        xds,
        weight_var=weight_var,
        interfaces=interfaces,
        use_nearest_interfaces=use_nearest_interfaces,
        keep_weight_var=keep_weight_var,
        _valid_mask=leading_wet,
    )

    mask = result["level"] > bottom_interface

    for var in result.data_vars:
        if "level" in result[var].dims and var != weight_var:
            result[var] = result[var].where(~mask)

    result["interface"].attrs.update(
        {
            "long_name": "resolved target layer interfaces",
            "units": xds["level"].attrs.get("units", "m"),
            "source_interface_definition": interface_source,
            "requested_interfaces": np.asarray(
                interfaces,
                dtype=np.float64,
            ).tolist(),
            "use_nearest_interfaces": bool(use_nearest_interfaces),
        }
    )
    result["level"].attrs.update(
        {
            "units": xds["level"].attrs.get("units", "m"),
            "positive": xds["level"].attrs.get("positive", "down"),
            "vertical_regrid_method": "native-interface center bins",
            "source_interface_definition": interface_source,
            "bottom_mask_policy": "target midpoint at or above z_i[N]",
            "missing_value_policy": "variable-valid weight normalization",
        }
    )
    if keep_weight_var:
        result[weight_var].attrs.update(
            {
                "represented_thickness": (
                    "nominal full target-bin thickness; not variable-valid "
                    "or locally wet thickness"
                ),
                "units": xds["level"].attrs.get("units", "m"),
            }
        )

    return result


def fv_vertical_regrid_ocn(*args, **kwargs) -> xr.Dataset:
    """Backward-compatible alias for the original ocean regrid name."""

    return fv_vertical_regrid_ocean(*args, **kwargs)
