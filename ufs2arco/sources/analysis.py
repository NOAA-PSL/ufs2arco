import logging
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.sources import Source

logger = logging.getLogger("ufs2arco")

class AnalysisSource(Source):
    """
    Base class for all deterministic analysis or reanalysis-like datasets.

    Child classes must map dimension names to:
        ``("time", "level", "latitude", "longitude")``
    """

    sample_dims = ("time",)
    base_dims = ("level", "latitude", "longitude")

    # fill these out per subclass
    static_vars = tuple()
    available_variables = tuple()
    available_levels = tuple()

    @property
    def rename(self) -> dict:
        """
        Use this to map to the dimensions
        (time, level, latitude, longitude)
        """
        return dict()

    def __init__(
        self,
        time: dict,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
    ) -> None:
        """
        Initialize the Source object.

        Args:
            time (dict): Dictionary with start and end times for the time vector, and e.g. "freq=6h". All options get passed to ``pandas.date_range``.
            variables (list, tuple, optional): variables to grab
            levels (list, tuple, optional): vertical levels to grab
            use_nearest_levels (bool, optional): if True, all level selection with
                ``xarray.Dataset.sel(level=levels, method="nearest")``
        """
        self.time = pd.date_range(**time)
        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
        )

    def open_sample_dataset(
        self,
        time: pd.Timestamp,
        open_static_vars: bool,
        cache_dir: Optional[str]=None,
    ) -> xr.Dataset:
        """
        Open a single dataset for the timestamp.

        The resulting dataset should always have dimensions:

        (time, level, latitude, longitude)

        Even if any one of those is single

        Args:
            time (pd.Timestamp): The timestamp.
            open_static_vars (bool): if True, read static_vars, otherwise don't
            cache_dir (str, optional): Directory to cache files locally.

        Returns:
            xr.Dataset: The dataset containing the specified data.
        """
        pass
