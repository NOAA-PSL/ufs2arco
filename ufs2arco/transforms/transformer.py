import logging

import xarray as xr

from ufs2arco.transforms.horizontal_regrid import horizontal_regrid
from ufs2arco.transforms.mappings import get_available_mappings, apply_mappings
from ufs2arco.transforms.rotate_vectors import rotate_vectors
from ufs2arco.transforms.vertical_regrid import fv_vertical_regrid

logger = logging.getLogger("ufs2arco")

class Transformer:

    @property
    def implemented(self) -> tuple:
        return (
            "multiply",
            "divide",
            "fv_vertical_regrid",
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

        if "rotate_vectors" in self.names:
            xds = rotate_vectors(xds, **self.options["rotate_vectors"])

        if "fv_vertical_regrid" in self.names:
            xds = fv_vertical_regrid(xds, **self.options["fv_vertical_regrid"])

        if "horizontal_regrid" in self.names:
            xds = horizontal_regrid(xds, **self.options["horizontal_regrid"])

        if "mappings" in self.names:
            xds = apply_mappings(xds, self.options["mappings"])

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
