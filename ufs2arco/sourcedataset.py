import logging
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("ufs2arco")

class SourceDataset:
    """Base class for all forecast-like datasets

    Note: it's kind of weird for all subclasses to take t0, fhr, and member as init inputs, but then require
    each subclass to state its sample_dims and base_dims, but whatever.
    """

    static_vars = tuple()
    sample_dims = tuple()
    base_dims = tuple()
    available_variables = tuple()
    available_levels = tuple()

    @property
    def rename(self) -> dict:
        """Use this to map to the dimensions
        (t0, fhr, member, level, latitude, longitude)
        """
        return dict()

    def __init__(
        self,
        t0: dict,
        fhr: dict,
        member: dict,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
    ) -> None:
        """
        Initialize the SourceDataset object.

        Args:
            t0 (dict): Dictionary with start and end times for initial conditions, and e.g. "freq=6h". All options get passed to ``pandas.date_range``.
            fhr (dict): Dictionary with 'start', 'end', and 'step' forecast hours.
            member (dict): Dictionary with 'start', 'end', and 'step' ensemble members.
            chunks (dict): Dictionary with chunk sizes for Dask arrays.
            variables (list, tuple, optional): variables to grab
            levels (list, tuple, optional): vertical levels to grab
        """
        self.t0 = pd.date_range(**t0)
        self.fhr = np.arange(fhr["start"], fhr["end"] + 1, fhr["step"])
        self.member = np.arange(member["start"], member["end"] + 1, member["step"])

        # check variable selection
        if variables is not None:
            unrecognized = []
            for v in variables:
                if v not in self.available_variables:
                    unrecognized.append(v)

            if len(unrecognized) > 0:
                raise NotImplementedError(f"{self.name}.__init__: the following variables are not recognized or not implemented: {unrecognized}")
            self.variables = variables
        else:
            self.variables = self.available_variables

        # check level selection
        if levels is not None:
            unrecognized = []
            for l in levels:
                if l not in self.available_levels:
                    unrecognized.append(l)

            if len(unrecognized) > 0:
                raise NotImplementedError(f"{self.name}.__init__: the following levels are not recognized or not implemented: {unrecognized}")
            self.levels = levels

        else:
            self.levels = self.available_levels

        logger.info(str(self))


    def __str__(self) -> str:
        """
        Return a string representation of the SourceDataset object.

        Returns:
            str: The string representation of the dataset.
        """
        title = f"Source: {self.name}"
        msg = f"\n{title}\n" + \
              "".join(["-" for _ in range(len(title))]) + "\n"
        for key in ["t0", "fhr", "member", "levels", "variables"]:
            msg += f"{key:<18s}: {getattr(self, key)}\n"
        return msg

    @property
    def name(self) -> str:
        return self.__class__.__name__


    def open_sample_dataset(
        self,
        t0: pd.Timestamp,
        fhr: int,
        member: int,
        cache_dir: str,
    ) -> xr.Dataset:
        """
        Open a single dataset for the given initial condition, forecast hour, and member.

        The resulting dataset should always have dimensions:

        (t0, fhr, member, level, latitude, longitude)

        Even if any one of those is single

        Args:
            t0 (pd.Timestamp): The initial condition timestamp.
            fhr (int): The forecast hour.
            member (int): The ensemble member ID.
            cache_dir (str): Directory to cache files locally.

        Returns:
            xr.Dataset: The dataset containing the specified data.
        """
        pass
