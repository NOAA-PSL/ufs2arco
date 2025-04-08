import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.sources import Source, NOAAGribForecastData

logger = logging.getLogger("ufs2arco")

class AWSGEFSArchive(NOAAGribForecastData, Source):
    """Access NOAA's forecast archive from the Global Ensemble Forecast System (GEFS) at s3://noaa-gefs-pds."""

    sample_dims = ("t0", "fhr", "member")
    horizontal_dims = ("latitude", "longitude")
    file_suffixes = ("a", "b")
    static_vars = ("lsm", "orog")

    @property
    def available_levels(self) -> tuple:
        return (
            10, 20, 30, 50, 70,
            100, 150, 200, 250, 300, 350, 400, 450,
            500, 550, 600, 650, 700, 750, 800, 850,
            900, 925, 950, 975, 1000,
        )

    @property
    def rename(self) -> dict:
        return {
            "time": "t0",
            "step": "lead_time",
            "isobaricInhPa": "level",
        }

    def __init__(
        self,
        t0: dict,
        fhr: dict,
        member: dict,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
    ) -> None:
        """
        Args:
            t0 (dict): Dictionary with start and end times for initial conditions, and e.g. "freq=6h". All options get passed to ``pandas.date_range``.
            fhr (dict): Dictionary with 'start', 'end', and 'step' forecast hours.
            member (dict): Dictionary with 'start', 'end', and 'step' ensemble members.
            variables (list, tuple, optional): variables to grab
            levels (list, tuple, optional): vertical levels to grab
            use_nearest_levels (bool, optional): if True, all level selection with
                ``xarray.Dataset.sel(level=levels, method="nearest")``
            slices (dict, optional): either "sel" or "isel", with slice, passed to xarray
        """
        self.t0 = pd.date_range(**t0)
        self.fhr = np.arange(fhr["start"], fhr["end"] + 1, fhr["step"])
        self.member = np.arange(member["start"], member["end"] + 1, member["step"])
        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
        )


    def _build_path(
        self,
        t0: pd.Timestamp,
        member: int,
        fhr: int,
        file_suffix: str,
    ) -> str:
        """
        Build the file path to a GEFS GRIB file on AWS.

        Args:
            t0 (pd.Timestamp): The initial condition timestamp.
            member (int): The ensemble member ID.
            fhr (int): The forecast hour.
            file_suffix (str): For GEFS, either 'a' or 'b'.

        Returns:
            str: The constructed file path.
        """
        c_or_p = "c" if member == 0 else "p"
        bucket = f"s3://noaa-gefs-pds"
        outer = f"gefs.{t0.year:04d}{t0.month:02d}{t0.day:02d}/{t0.hour:02d}"
        # Thanks to Herbie for figuring these out
        if t0 < pd.Timestamp("2018-07-27"):
            middle = ""
            fname = f"ge{c_or_p}{member:02d}.t{t0.hour:02d}z.pgrb2{file_suffix}f{fhr:03d}"
        elif t0 < pd.Timestamp("2020-09-23T12"):
            middle = f"pgrb2{file_suffix}/"
            fname = f"ge{c_or_p}{member:02d}.t{t0.hour:02d}z.pgrb2{file_suffix}f{fhr:02d}"
        else:
            middle = f"atmos/pgrb2{file_suffix}p5/"
            fname = f"ge{c_or_p}{member:02d}.t{t0.hour:02d}z.pgrb2{file_suffix}.0p50.f{fhr:03d}"
        fullpath = f"filecache::{bucket}/{outer}/{middle}{fname}"
        logger.debug(f"{self.name}._build_path: reading {fullpath}")
        return fullpath
