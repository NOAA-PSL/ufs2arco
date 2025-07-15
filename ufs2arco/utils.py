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


def expand_anemoi_dataset(ads: xr.Dataset, name: str, variable_names: list[str]) -> xr.Dataset:
    """Expand the anemoi dataset "variable" dimension to individual variables.

    Converts a DataArray within the anemoi dataset,
    with any subset of dimensions [time, variable, ensemble, cell] into an xarray.Dataset
    where each variable has dimensions [time, cell] for 2D variables, or [time, level, cell] for 3D variables.

    Note that this renames "latitudes" to "latitude", "longitudes" to "longitude", and swaps the logical "time"
    dimension with "dates", and renames "dates" to "time".

    Example:
        >>> import xarray as xr
        >>> from ufs2arco.utils import expand_anemoi_dataset
        >>> ads = xr.open_zarr("/path/to/my-anemoi-dataset.zarr")
        >>> the_data = expand_anemoi_dataset(ads, "data", ads.attrs["variables"])
        >>> maximum_values = expand_anemoi_dataset(ads, "maximum", ads.attrs["variables"])

    Parameters:
        ads (xr.Dataset): anemoi dataset, opened with xarray
        name (str): name of the array within the anemoi dataset to expand, e.g. "data" or any of the statistics ("maximum", "minimum", "squares", "mean", etc)
        variable_names (list[str]): A list of variable names corresponding to the 'variable' dimension.

    Returns:
        xr.Dataset: A dataset with each variable as a [time, cell] DataArray.
    """
    xda = ads[name]
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

    # rename stuff, set coordinates, swap dims
    for key in ["dates", "latitudes", "longitudes"]:
        xds[key] = ads[key]
        xds = xds.set_coords(key)
    xds = xds.rename({"latitudes": "latitude", "longitudes": "longitude"})
    xds = xds.swap_dims({"time": "dates"}).drop_vars("time").rename({"dates": "time"})
    return xds


def convert_anemoi_inference_dataset(xds: xr.Dataset):
    """Convert the output from anemoi-inference to a multivariate xarray dataset.

    Stack each variable separated by vertical level to a single variable with a level dimension

    Note that this also renames dimension names "values" -> "cell" so that it is consistent with
    the anemoi dataset naming convention, and to avoid issues with xarray.DataArray.values.

    Example:
        >>> import xarray as xr
        >>> from ufs2arco.utils import convert_anemoi_inference_dataset
        >>> ids = xr.open_dataset("/path/to/anemoi_inference.nc", decode_timedelta=True, chunks="auto")
        >>> convert_anemoi_inference_dataset(ids)

    Args:
        xds (xr.Dataset): result from anemoi inference run, stored as netcdf

    Returns:
        result (xr.Dataset): with vertical levels stacked and "values" renamed to "cell"
    """

    # Use regex to detect base name and level
    level_pattern = re.compile(r"^(?P<base>[a-zA-Z0-9]+)_(?P<level>\d+)$")
    level_groups = defaultdict(list)
    flat_vars = []

    for name in xds.data_vars:
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
            this_2d_array = xds[level_name].expand_dims({"level": [level]})
            attrs = xds[level_name].attrs.copy()
            dalist.append(this_2d_array)

        dsdict[base] = xr.concat(dalist, dim="level")
        dsdict[base].attrs = attrs

    # Add flat variables
    for name in flat_vars:
        dsdict[name] = xds[name]
        dsdict[name].attrs = xds[name].attrs.copy()

    result = xr.Dataset({k: dsdict[k] for k in sorted(dsdict)}, attrs=xds.attrs)
    result = result.set_coords(["latitude", "longitude"])
    result = result.rename({"values": "cell"})
    for key in result.data_vars:
        dims = tuple(d for d in ("time", "ensemble", "level", "cell") if d in result[key].dims)
        result[key] = result[key].transpose(*dims)
    return result
