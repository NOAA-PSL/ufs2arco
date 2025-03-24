import logging
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.sources import Source

logger = logging.getLogger("ufs2arco")

class EnsembleForecastSource(Source):
    """
    Base class for all ensemble forecast-like datasets
    """

    sample_dims = ("t0", "fhr", "member")
    base_dims = ("level", "latitude", "longitude")

    # fill these out per subclass
    static_vars = tuple()
    available_variables = tuple()
    available_levels = tuple()

    @property
    def rename(self) -> dict:
        """
        Use this to map to the dimensions
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
        Initialize the Source object.

        Args:
            t0 (dict): Dictionary with start and end times for initial conditions, and e.g. "freq=6h". All options get passed to ``pandas.date_range``.
            fhr (dict): Dictionary with 'start', 'end', and 'step' forecast hours.
            member (dict): Dictionary with 'start', 'end', and 'step' ensemble members.
            variables (list, tuple, optional): variables to grab
            levels (list, tuple, optional): vertical levels to grab
        """
        self.t0 = pd.date_range(**t0)
        self.fhr = np.arange(fhr["start"], fhr["end"] + 1, fhr["step"])
        self.member = np.arange(member["start"], member["end"] + 1, member["step"])
        super().__init__(
            variables=variables,
            levels=levels,
        )


    def __str__(self) -> str:
        title = f"Source: {self.name}"
        msg = f"\n{title}\n" + \
              "".join(["-" for _ in range(len(title))]) + "\n"
        for key in ["t0", "fhr", "member", "variables", "levels"]:
            msg += f"{key:<18s}: {getattr(self, key)}\n"
        return msg


    def open_sample_dataset(
        self,
        t0: pd.Timestamp,
        fhr: int,
        member: int,
        cache_dir: Optional[str]=None,
        open_static_vars: Optional[bool]=True,
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


    def add_full_extra_coords(self, xds: xr.Dataset) -> xr.Dataset:
        # compute the full version of these extra coords
        xds["lead_time"] = xr.DataArray(
            [pd.Timedelta(hours=x) for x in self.fhr],
            coords=xds["fhr"].coords,
            attrs=xds["lead_time"].attrs.copy(),
        )

        xds["valid_time"] = xds["t0"] + xds["lead_time"]
        xds["valid_time"].attrs = xds["valid_time"].attrs.copy()
        xds = xds.set_coords(["lead_time", "valid_time"])
        return xds
