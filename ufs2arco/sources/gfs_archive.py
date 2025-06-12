import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.sources import Source, NOAAGribForecastData

logger = logging.getLogger("ufs2arco")

class GFSArchive(NOAAGribForecastData, Source):
    """
    Access 1/4 degree archives of NOAA's Global Forecast System (GFS) via:
        * if before 2021: UCAR Research Data Archive (RDA)
            * https://rda.ucar.edu/datasets/d084001
            * https://rda.ucar.edu/datasets/d084003
        * after 2021: AWS at https://registry.opendata.aws/noaa-gfs-bdp-pds/
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


        # make sure the default variable set "just works"
        # by pulling out the variable names only available at forecast time
        if tuple(self.variables) == self.available_variables:
            varlist = []
            for key, meta in self._varmeta.items():
                if not self._varmeta[key]["forecast_only"]:
                    varlist.append(key)
            self.variables = varlist

        # for GFS, plenty of variables only exist in the forecast, not analysis grib files
        # make sure the user doesn't ask for these before we get started
        if any(self.fhr == 0):
            requested_vars_not_in_analysis = []
            for varname in self.variables:
                if self._varmeta[varname]["forecast_only"]:
                    requested_vars_not_in_analysis.append(varname)
            if len(requested_vars_not_in_analysis) > 0:
                msg = f"{self.name}.__init__: the following requested variables only exist in forecast timesteps"
                msg += f"\n\n{requested_vars_not_in_analysis}\n\n"
                msg += "these should be requested separately with only fhr > 0 (i.e., fhr start >0 in your yaml)"
                raise Exception(msg)

    def _build_path(
        self,
        t0: pd.Timestamp,
        fhr: int,
        file_suffix: str,
    ) -> str:
        """
        Build the file path to a GFS GRIB file on RDA or AWS.

        Args:
            t0 (pd.Timestamp): The initial condition timestamp.
            fhr (int): The forecast hour.
            file_suffix (str): For GFS, either '' or 'b'.

        Returns:
            str: The constructed file path.
        """
        if t0 < pd.Timestamp("2021-01-01T00"):

            bucket = f"https://data.rda.ucar.edu/d084001" if file_suffix == "" else \
                    f"https://data.rda.ucar.edu/d084003"
            outer = f"{t0.year:04d}/{t0.year:04d}{t0.month:02d}{t0.day:02d}"
            fname = f"gfs.0p25{file_suffix}.{t0.year:04d}{t0.month:02d}{t0.day:02d}{t0.hour:02d}.f{fhr:03d}.grib2"


        else:
            bucket = "s3://noaa-gfs-bdp-pds"
            outer = f"gfs.{t0.year:04d}{t0.month:02d}{t0.day:02d}/{t0.hour:02d}"
            fname = f"gfs.t{t0.hour:02d}z.pgrb2{file_suffix}.0p25.f{fhr:03d}"

            if t0 > pd.Timestamp("2021-03-22T06"):
                fname = "atmos/" + fname

        fullpath = f"filecache::{bucket}/{outer}/{fname}"
        logger.debug(f"{self.name}._build_path: reading {fullpath}")
        return fullpath
