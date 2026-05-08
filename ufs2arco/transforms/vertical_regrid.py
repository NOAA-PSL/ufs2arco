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

    z_i = np.array(
        [
            0.0,
            1.0308075249195099,
            2.1117684841156006,
            3.2619810104370117,
            4.498320460319519,
            5.841697454452515,
            7.317776441574097,
            8.957634449005127,
            10.79821491241455,
            12.882359981536865,
            15.258180141448975,
            17.97756004333496,
            21.09370994567871,
            24.657959938049316,
            28.716429710388184,
            33.307379722595215,
            38.46010971069336,
            44.19595527648926,
            50.53126525878906,
            57.48173904418945,
            65.0672607421875,
            73.31642150878906,
            82.27029418945312,
            91.98537063598633,
            102.53580474853516,
            114.0151481628418,
            126.53790283203125,
            140.24100494384766,
            155.2853546142578,
            171.8572006225586,
            190.1699447631836,
            210.4655990600586,
            233.01589965820312,
            258.1235046386719,
            286.12196350097656,
            317.3751525878906,
            352.27565002441406,
            391.2412109375,
            434.7097625732422,
            483.13275146484375,
            536.9660949707031,
            596.6591491699219,
            662.6423645019531,
            735.3135070800781,
            815.0236511230469,
            902.0634460449219,
            996.6509399414062,
            1098.9224853515625,
            1208.9259643554688,
            1326.6189575195312,
            1451.8704833984375,
            1584.4660034179688,
            1724.1195068359375,
            1870.4835205078125,
            2023.1615600585938,
            2181.724609375,
            2345.7235107421875,
            2514.702392578125,
            2688.20947265625,
            2865.8045654296875,
            3047.0675048828125,
            3231.60302734375,
            3419.0435791015625,
            3609.051513671875,
            3801.3194580078125,
            3995.5704345703125,
            4191.5560302734375,
            4389.053955078125,
            4587.867919921875,
            4787.825439453125,
            4988.7744140625,
            5190.5810546875,
            5393.129150390625,
            5596.318115234375,
            5800.0595703125,
            6004.056640625,
        ]
    )

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
