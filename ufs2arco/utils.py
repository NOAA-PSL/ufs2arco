import itertools
import re
from collections import defaultdict

import numpy as np
import xarray as xr

def batched(iterable, n):
    """A homegrown version of itertools.batched, taken
    directly from `itertools docs <https://docs.python.org/3/library/itertools.html#itertools.batched>`_
    for earlier versions of itertools

    Args:
        iterable (Iterable): loop over this
        n (int): batch size

    Returns:
        batch (Iterable): a batch of size n from the iterable
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def expand_anemoi_to_dataset(xda: xr.DataArray, variable_names: list[str]) -> xr.Dataset:
    """
    Converts a DataArray with dimensions [time, variable, cell] into an xarray.Dataset
    where each variable has dimensions [time, cell].

    Parameters:
        xda (xr.DataArray): The input data array with dimensions [time, variable, cell].
        variable_names (list[str]): A list of variable names corresponding to the 'variable' dimension.

    Returns:
        xr.Dataset: A dataset with each variable as a [time, cell] DataArray.
    """
    if xda.sizes['variable'] != len(variable_names):
        raise ValueError("Length of variable_names does not match the size of the 'variable' dimension.")

    # Use regex to detect base name and level
    level_pattern = re.compile(r"^(?P<base>[a-zA-Z0-9]+)_(?P<level>\d+)$")
    level_groups = defaultdict(list)
    flat_vars = []

    for name in variable_names:
        parts = name.rsplit("_", 1)
        if len(parts) > 1 and parts[-1].isdigit():
            base = parts[0]
            level = int(parts[-1])
            level_groups[base].append((level, name))
        else:
            flat_vars.append(name)

    dsdict = {}

    # Add grouped variables with levels
    for base, level_name_pairs in level_groups.items():

        dalist = []
        for level, level_name in level_name_pairs:
            this_2d_array = xda.sel(variable=variable_names.index(level_name))
            this_2d_array = this_2d_array.drop_vars("variable")
            this_2d_array = this_2d_array.expand_dims({"level": [level]})
            dalist.append(this_2d_array)

        dsdict[base] = xr.concat(dalist, dim="level")

    # Add flat variables
    for name in flat_vars:
        dsdict[name] = xda.sel(variable=variable_names.index(name)).drop_vars("variable")

    xds = xr.Dataset({k: dsdict[k] for k in sorted(dsdict)})
    for key in xds.data_vars:
        dims = tuple(d for d in ("time", "ensemble", "level", "cell") if d in xds[key].dims)
        xds[key] = xds[key].transpose(*dims)
    return xds
