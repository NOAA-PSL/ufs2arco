import logging

import pandas as pd
import xarray as xr

from ufs2arco.targetdataset import TargetDataset

logger = logging.getLogger("ufs2arco")

class AnemoiDataset(TargetDataset):
    """
    Store dataset ready for anemoi

    Expected output has dimensions
        ("time", "variable", "ensemble", "cell"")

    Use the rename argument to modify any of these (and make sure :attr:`chunks` uses those desired names too),
    but note this might cause problems with anemoi!
    """

    sample_dims = ("time", "ensemble")
    base_dims = ("variable", "cell")

    @property
    def time(self):
        #TODO: this is the hack where I assume we're using fhr=0 always
        return self.source.t0 + pd.Timedelta(hours=self.source.fhr[0])

    @property
    def ensemble(self):
        return self.source.member

    def manage_coords(self, xds: xr.Dataset) -> xr.Dataset:
        return xds
