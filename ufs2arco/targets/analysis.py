import logging
from typing import Optional

import pandas as pd
import xarray as xr

from ufs2arco.sources import Source
from ufs2arco.targets import Target

logger = logging.getLogger("ufs2arco")

class Analysis(Target):
    """
    Sort of a default target format.
    Store the data in zarr, essentially how it's laid out originally

    Expected output has dimensions
        ("time", "level", "latitude", "longitude")

    Use the rename argument to modify any of these (and make sure :attr:`chunks` uses those desired names too)
    """

    sample_dims = ("time",)
    base_dims = ("level", "latitude", "longitude")

    # Each sample dim needs to be provided here
    # this is a simple case, target is passive to source
    @property
    def time(self):
        return self.source.time
