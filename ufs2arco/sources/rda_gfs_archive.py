import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.sources import Source, NOAAGribForecastData

logger = logging.getLogger("ufs2arco")

class RDAGFSArchive(NOAAGribForecastData, Source):
    """
    Access the UCAR Research Data Archive (RDA) of NOAA's Global Forecast System (GFS).

    For more see:
        * https://rda.ucar.edu/datasets/d084001
        * https://rda.ucar.edu/datasets/d084003

    Note:
        I don't know if the resolution changes at some point, as it does in the GEFS archive.
    """

    sample_dims = ("t0", "fhr")
    horizontal_dims = ("latitude", "longitude")
    file_suffixes = ("", "b")
    static_vars = ("lsm", "orog")

    @property
    def available_levels(self) -> tuple:
        return (
            1, 2, 3, 5, 7, 10, 15,
            20, 30, 40, 50, 70,
            100, 125, 150, 175, 200, 225, 250, 275,
            300, 325, 350, 375, 400, 425, 450, 475,
            500, 525, 550, 575, 600, 625, 650, 675,
            700, 725, 750, 775, 800, 825, 850, 875,
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
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
    ) -> None:
        """
        Args:
            t0 (dict): Dictionary with start and end times for initial conditions, and e.g. "freq=6h". All options get passed to ``pandas.date_range``.
            fhr (dict): Dictionary with 'start', 'end', and 'step' forecast hours.
            variables (list, tuple, optional): variables to grab
            levels (list, tuple, optional): vertical levels to grab
            use_nearest_levels (bool, optional): if True, all level selection with
                ``xarray.Dataset.sel(level=levels, method="nearest")``
            slices (dict, optional): either "sel" or "isel", with slice, passed to xarray
        """
        self.t0 = pd.date_range(**t0)
        self.fhr = np.arange(fhr["start"], fhr["end"] + 1, fhr["step"])
        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
        )


    def _build_path(
        self,
        t0: pd.Timestamp,
        fhr: int,
        file_suffix: str,
    ) -> str:
        """
        Build the file path to a GEFS GRIB file on AWS.

        Args:
            t0 (pd.Timestamp): The initial condition timestamp.
            fhr (int): The forecast hour.
            file_suffix (str): For GFS, either '' or 'b'.

        Returns:
            str: The constructed file path.
        """
        bucket = f"https://data.rda.ucar.edu/d084001" if file_suffix == "" else \
                f"https://data.rda.ucar.edu/d084003"
        outer = f"{t0.year:04d}/{t0.year:04d}{t0.month:02d}{t0.day:02d}"
        fname = f"gfs.0p25{file_suffix}.{t0.year:04d}{t0.month:02d}{t0.day:02d}{t0.hour:02d}.f{fhr:03d}.grib2"
        fullpath = f"filecache::{bucket}/{outer}/{fname}"
        logger.debug(f"{self.name}._build_path: reading {fullpath}")
        return fullpath
