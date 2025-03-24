import logging
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.sources import Source

logger = logging.getLogger("ufs2arco")

class AnalysisSource(Source):
    """
    Base class for all deterministic analysis or reanalysis-like datasets
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
    ) -> None:
        """
        Initialize the Source object.

        Args:
            time (dict): Dictionary with start and end times for the time vector, and e.g. "freq=6h". All options get passed to ``pandas.date_range``.
            variables (list, tuple, optional): variables to grab
            levels (list, tuple, optional): vertical levels to grab
        """
        self.time = pd.date_range(**time)
        super().__init__(
            variables=variables,
            levels=levels,
        )


    def __str__(self) -> str:
        """
        Return a string representation of the Source object.

        Returns:
            str: The string representation of the dataset.
        """
        title = f"Source: {self.name}"
        msg = f"\n{title}\n" + \
              "".join(["-" for _ in range(len(title))]) + "\n"
        for key in ["time", "variables", "levels"]:
            msg += f"{key:<18s}: {getattr(self, key)}\n"
        return msg


    def open_sample_dataset(
        self,
        time: pd.Timestamp,
        cache_dir: Optional[str]=None,
        open_static_vars: Optional[bool]=True
    ) -> xr.Dataset:
        """
        Open a single dataset for the timestamp.

        The resulting dataset should always have dimensions:

        (time, level, latitude, longitude)

        Even if any one of those is single

        Args:
            time (pd.Timestamp): The timestamp.
            cache_dir (str, optional): Directory to cache files locally.
            open_static_vars (bool, optional): If True, return dataset with non-time-varying variables, otherwise don't include them

        Returns:
            xr.Dataset: The dataset containing the specified data.
        """
        pass
