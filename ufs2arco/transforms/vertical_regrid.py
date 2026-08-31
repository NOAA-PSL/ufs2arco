import logging
from typing import Optional

import numpy as np
import os
import xarray as xr
import yaml

try:
    import flox
    _has_flox = True
except ImportError:
    _has_flox = False

logger = logging.getLogger("ufs2arco")

def fv_vertical_regrid(
    xds: xr.Dataset,
    weight_var: str,
    interfaces: list | np.ndarray,
    use_nearest_interfaces: Optional[bool] = False,
    keep_weight_var: Optional[bool] = False,
) -> xr.Dataset:
    """Vertically regrid a dataset based on interface values

    Args:
        xds (xr.Dataset)
        weight_var (str): with the name of the variable to weight regridding by
        interfaces (array_like): array of interface values to average vertical coordinate "level" between
            if "interface" does not exist in the dataset, this will be added, otherwise we subsample
        use_nearest_interfaces (bool, optional): if True, and "interface" is in the dataset,
            grab interfaces from dataset via
            ``xarray.Dataset.sel(interface=interfaces, method='nearest')``
        keep_weight_var (bool, optional): if False, drop the weight variable from the dataset before saving

    Returns:
        xds (xr.Dataset): with vertical averaging
    """

    if not _has_flox:
        logger.warning("Could not import flox, install with 'conda install -c conda-forge flox' for faster volume averaging (i.e. groupby operations)")

    assert weight_var in xds, \
        f"fv_vertical_regrid: can't find {weight_var} in dataset, can't use it for regridding"

    # get or create the exact interface values
    if "interface" in xds:
        kw = {"method": "nearest"} if use_nearest_interfaces else {}
        try:
            xds = xds.sel(interface=interfaces, **kw)
        except ValueError:
            msg = f"fv_vertical_regrid: couldn't find specified interfaces"
            msg += "\nSet option 'use_nearest_interfaces=True' if you want to use xarray.Dataset.sel(interface=interfaces, method='nearest')"
            raise ValueError(msg)

    else:
        xds["interface"] = xr.DataArray(
            interfaces,
            coords={"interface": interfaces},
            dims="interface",
            attrs={"description": "vertical coordinate interfaces created for regridding"},
        )

    # get the regrid weighting
    layer_thickness = xds[weight_var].groupby_bins(
        "level",
        bins=xds["interface"],
    ).sum()
    layer_thickness.attrs = xds[weight_var].attrs.copy()
    layer_thickness_inverse = 1/layer_thickness

    # do the regridding
    vars3d = [x for x in xds.data_vars if "level" in xds[x].dims and x != weight_var]
    for key in vars3d:
        attrs = xds[key].attrs.copy()
        xds[key] = layer_thickness_inverse * (
            (
                xds[key]*xds[weight_var]
            ).groupby_bins(
                "level",
                bins=xds["interface"],
            ).sum()
        )
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


def fv_vertical_regrid_ocean(
    xds: xr.Dataset,
    interfaces: list | np.ndarray,
    weight_var: Optional[str] = "dz",
    use_nearest_interfaces: Optional[bool] = True,
    keep_weight_var: Optional[bool] = False,
) -> xr.Dataset:
    """Vertically regrid ocean data and mask regridded layers below bathymetry.

    This computes MOM6 layer thickness from the replay ocean vertical
    interfaces, uses that thickness as the finite-volume weight, then masks
    variables below the last valid source layer in each horizontal column.
    """

    # Resolve relative to the package, not the repo root, so this works from an
    # installed copy. The file lives next to replay_vertical_levels.yaml, its
    # atmospheric counterpart.
    cfg_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "replay_ocean_vertical_levels.yaml",
        )
    )
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    z_i = np.array(cfg["z_i"])

    if "temp" not in xds:
        raise KeyError("fv_vertical_regrid_ocean: expected 'temp' in dataset to infer ocean bottom mask")

    valid_mask = ~xds["temp"].isnull()
    num_valid_layers = valid_mask.sum(dim="level")
    index = (num_valid_layers - 1).clip(min=0, max=len(z_i) - 1).astype(int)

    bottom_interface = xr.apply_ufunc(
        lambda idx: z_i[idx],
        index,
        vectorize=True,
        input_core_dims=[[]],
        output_dtypes=[float],
        dask="parallelized" if index.chunks else None,
    )
    bottom_interface.name = "bottom_interface"
    bottom_interface = bottom_interface.assign_coords(
        {
            "latitude": xds["latitude"],
            "longitude": xds["longitude"],
        }
    )

    dz = np.diff(z_i)
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

    result = fv_vertical_regrid(
        xds,
        weight_var=weight_var,
        interfaces=interfaces,
        use_nearest_interfaces=use_nearest_interfaces,
        keep_weight_var=keep_weight_var,
    )

    mask = result["level"] > bottom_interface

    for var in result.data_vars:
        if "level" in result[var].dims:
            result[var] = result[var].where(~mask)

    return result


def fv_vertical_regrid_ocn(*args, **kwargs) -> xr.Dataset:
    """Backward-compatible alias for the original ocean regrid name."""

    return fv_vertical_regrid_ocean(*args, **kwargs)
