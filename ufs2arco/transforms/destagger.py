import logging
from typing import Optional

import numpy as np
import xarray as xr
import xwrf

logger = logging.getLogger("ufs2arco")

def destagger(
    xds: xr.Dataset,
    variable: str,
    dimension: str,
) -> xr.Dataset:
    """Destagger dataset along staggered dimensions

    Args:
        xds (xr.Dataset)
        variable (str): the name of the variable to destagger
        dimension(str): dimension to destagger

    Returns:
        xds (xr.Dataset): with vertical averaging
    """

    """apply destaggering of grid points"""

    #dims = list(xds.dims)
    #variable_index = dims.index(dimension)
    #stagger_dim = variable_index - len(dims) 

    xds = xds.rename({dimension: f"{dimension}_stag"})

    # Destagger the entire dataset or specific variables to the mass grid
    xds[variable] = xds[variable].xwrf.destagger()
    return xds
