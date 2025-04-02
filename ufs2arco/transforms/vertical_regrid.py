import logging
from typing import Optional

import numpy as np
import xarray as xr

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
