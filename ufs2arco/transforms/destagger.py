import logging

import numpy as np
import xarray as xr
import xwrf

logger = logging.getLogger("ufs2arco")

def destagger(
    xds: xr.Dataset,
    config: dict,
) -> xr.Dataset:
    """Destagger specified variables along staggered dimension(s)

    Args:
        xds (xr.Dataset)
        config (dict): with pattern
            {variable: dimension} where
                variable (str): the name of the variable to destagger
                dimension(str): dimension to destagger

    Returns:
        xds (xr.Dataset): with destaggering applied to each specified variable & dimension pair
    """

    # Loop through all variables and their staggered dimensions - as specified in recipe yaml
    for variable, dimension in config.items():
        # if target staggered dimension in the dataset, proceed with destaggering it
        if f"{dimension}_stag" in xds[variable].dims:
            xds[variable] = xds[variable].xwrf.destagger()           
        else:
            logger.info(f"Destagger could not be applied to variable {variable}.\n    Expected {dimension}_stag dimension not present in dataset.")
    
    # Clean up any lingering staggered dimensions that are no longer in use
    used_dims = set()
    for var in xds.data_vars.values():
        used_dims.update(var.dims)
    
    # All dataset dimensions
    all_dims = set(xds.dims)
    # Isolate just the unused dimensions
    unused_dims = all_dims - used_dims
    # Drop unused dimensions and associated coordinates
    xds = xds.drop_dims(list(unused_dims),errors="ignore")

    return xds
