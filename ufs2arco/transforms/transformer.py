import logging

import xarray as xr

from ufs2arco.transforms.vertical_regrid import fv_vertical_regrid

logger = logging.getLogger("ufs2arco")

class Transformer:

    @property
    def implemented(self) -> tuple:
        return (
            "fv_vertical_regrid",
        )

    def __init__(self, options):

        names = list(options.keys())

        unrecognized = []
        for name in names:
            if name not in self.implemented:
                unrecognized.append(name)

        if len(unrecognized) > 0:
            raise NotImplementedError(f"Transformer.__init__: the following transformations are not recognized or not implemented: {unrecognized}")

        self.names = names
        self.options = options

    def __call__(self, xds: xr.Dataset):
        """
        Process the dataset, performing any of the desired transformations

        Args:
            xds (xr.Dataset): with the data

        Returns:
            xds (xr.Dataset): the processed version
        """

        if "fv_vertical_regrid" in self.names:
            xds = fv_vertical_regrid(xds, **self.options["fv_vertical_regrid"])

        return xds

