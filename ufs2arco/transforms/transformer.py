import logging

import xarray as xr

from ufs2arco.transforms.horizontal_regrid import horizontal_regrid
from ufs2arco.transforms.mappings import get_available_mappings, apply_mappings
from ufs2arco.transforms.ocean import mixed_layer_depth, ocean_density, ocean_heat_content
from ufs2arco.transforms.rotate_vectors import rotate_vectors
from ufs2arco.transforms.vertical_regrid import fv_vertical_regrid
from ufs2arco.transforms.vertical_regrid import fv_vertical_regrid_ocn
from ufs2arco.transforms.vertical_regrid import fv_vertical_regrid_ocean

logger = logging.getLogger("ufs2arco")

class Transformer:

    @property
    def implemented(self) -> tuple:
        return (
            "multiply",
            "divide",
            "rename",
            "ocean_density",
            "mixed_layer_depth",
            "ocean_heat_content",
            "fv_vertical_regrid",
            "fv_vertical_regrid_ocn",
            "fv_vertical_regrid_ocean",
            "horizontal_regrid",
            "mappings",
            "rotate_vectors",
        )

    def __init__(self, options):

        names = list(options.keys())

        # first check regrid, mappings, etc
        unrecognized = []
        for name in names:
            if name not in self.implemented:
                unrecognized.append(name)

        if len(unrecognized) > 0:
            raise NotImplementedError(f"Transformer.__init__: the following transformations are not recognized or not implemented: {unrecognized}")

        # now check for mappings
        unrecognized = []
        if "mappings" in names:
            available = list(get_available_mappings().keys())
            for mapname in options["mappings"]:
                if mapname not in available:
                    unrecognized.append(mapname)

            if len(unrecognized) > 0:
                raise NotImplementedError(f"Transformer.__init__: the following mappings are not recognized or not implemented: {unrecognized}")

        # if we want to do horizontal regridding, check if xesmf is installed
        if "horizontal_regrid" in names:
            try:
                import xesmf
            except ImportError:
                raise ImportError(f"Transformer.__init__: Could not 'import xesmf', but this is needed for 'horizontal_grid' transformations. Install xesmf with\n'conda install -c conda-forge xesmf'")

        self.names = names
        self.options = options
        logger.info(str(self))

    def __str__(self) -> str:
        title = f"Transformations"
        msg = f"\n{title}\n" + \
              "".join(["-" for _ in range(len(title))]) + "\n"
        optstr = "\n    ".join([f"{key:<14s}: {val}" for key, val in self.options.items()])
        msg += f"options\n    {optstr}\n"
        return msg

    def __call__(self, xds: xr.Dataset):
        """
        Process the dataset, performing any of the desired transformations

        Args:
            xds (xr.Dataset): with the data

        Returns:
            xds (xr.Dataset): the processed version
        """

        if "multiply" in self.names:
            xds = multiply(xds, self.options["multiply"])

        if "divide" in self.names:
            xds = divide(xds, self.options["divide"])

        # Ocean diagnostics run here, before any vertical regridding, so they see
        # the native water column. The mixed layer lives in the top tens of
        # meters, so computing it on coarsened layers would flatten the field.
        if "ocean_density" in self.names:
            xds = ocean_density(xds, **self.options["ocean_density"])

        if "mixed_layer_depth" in self.names:
            xds = mixed_layer_depth(xds, **self.options["mixed_layer_depth"])

        if "ocean_heat_content" in self.names:
            xds = ocean_heat_content(xds, **self.options["ocean_heat_content"])

        if "rotate_vectors" in self.names:
            xds = rotate_vectors(xds, **self.options["rotate_vectors"])

        if "fv_vertical_regrid" in self.names:
            xds = fv_vertical_regrid(xds, **self.options["fv_vertical_regrid"])

        if "fv_vertical_regrid_ocn" in self.names:
            xds = fv_vertical_regrid_ocn(xds, **self.options["fv_vertical_regrid_ocn"])

        if "fv_vertical_regrid_ocean" in self.names:
            xds = fv_vertical_regrid_ocean(xds, **self.options["fv_vertical_regrid_ocean"])

        if "horizontal_regrid" in self.names:
            xds = horizontal_regrid(xds, **self.options["horizontal_regrid"])

        if "mappings" in self.names:
            xds = apply_mappings(xds, self.options["mappings"])

        if "rename" in self.names:
            xds = rename(xds, self.options["rename"])

        return xds

def multiply(xds, config):
    """
    Multiply selected variables by a scalar, or if it can be provided in a yaml, an array of
    an appropriately broadcastable size

    Note that for now, attrs are not preserved in this process, for hopefully obvious reasons

    Args:
        xds (xr.Dataset): the dataset from source
        config (dict): with pattern
            {varname: scalar_value_to_multiply_by}

    Returns:
        xds (xr.Dataset): with new dataset
    """
    for varname, scalar in config.items():
        if varname in xds:
            xds[varname] = xds[varname] * scalar
    return xds

def divide(xds, config):
    """
    Divide selected variables by a scalar, or if it can be provided in a yaml, an array of
    an appropriately broadcastable size

    Note that for now, attrs are not preserved in this process, for hopefully obvious reasons

    Args:
        xds (xr.Dataset): the dataset from source
        config (dict): with pattern
            {varname: scalar_value_to_divide_by}

    Returns:
        xds (xr.Dataset): with new dataset
    """
    for varname, scalar in config.items():
        if varname in xds:
            xds[varname] = xds[varname] / scalar
    return xds

def rename(xds, config):
    """
    Rename variables, coordinates, or dimensions in the dataset.

    Args:
        xds (xr.Dataset): the dataset from source
        config (dict): with pattern {old_name: new_name}

    Returns:
        xds (xr.Dataset): with requested names changed
    """
    rename_map = {}
    known_names = set(xds.variables) | set(xds.dims)
    for old_name, new_name in config.items():
        if old_name not in known_names:
            logger.info(f"rename: {old_name} not found in dataset, skipping.")
        elif new_name in known_names and new_name != old_name:
            raise ValueError(
                f"rename: can't rename {old_name} to {new_name}; {new_name} already exists."
            )
        else:
            rename_map[old_name] = new_name

    if len(rename_map) > 0:
        xds = xds.rename(rename_map)
    return xds
