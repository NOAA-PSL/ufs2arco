import logging

import numpy as np
import xarray as xr

logger = logging.getLogger("ufs2arco")

def get_available_mappings() -> dict:
    return {
        "log": _log,
        "round": _round,
    }

def apply_mappings(
    xds: xr.Dataset,
    mappings: dict,
) -> xr.Dataset:
    """
    Apply any of the recognized mappings listed in :func:`get_available_mappings`

    Args:
        xds (xr.Dataset): with pre-mapped data
        mappings (dict): with a dictionary with keys as function names, and list of variables to apply it to, e.g.
            {"log": ["spfh", "spfh2m"]}

    Returns:
        xds (xr.Dataset): with mapped data, and originally mapped fields removed
    """

    function_mappings = get_available_mappings()

    for mapname, varlist in mappings.items():
        varlist = [varlist] if isinstance(varlist, str) else varlist
        func = function_mappings[mapname]
        for varname in varlist:
            if varname in xds:
                new_name = f"{mapname}_{varname}"
                xds[new_name] = func(xds[varname])
                long_name = xds[varname].attrs.get("long_name", varname)
                xds[new_name].attrs["long_name"] = f"{mapname} of {long_name}"
                xds = xds.drop_vars(varname)
    return xds

def _log(xda):
    """Performs a protected log of the provided array"""
    cond = xda > 0
    lda = xr.where(
        cond,
        np.log(xda.where(cond)),
        0.,
    )

    # unsure if this is truly what we should do... but whatever
    lda.attrs = xda.attrs.copy()
    lda.attrs["units"] = ""
    return lda

def _round(xda):
    """apply numpy.round"""
    with xr.set_options(keep_attrs=True):
        return np.round(xda)

