import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.sources import Source, NOAAGribForecastData

logger = logging.getLogger("ufs2arco")


class GDASArchive(NOAAGribForecastData, Source):
    """
    Access 0.25 or 1.0 degree archives of NOAA's Global Forecast System (GFS) via AWS:
        * t0 > pd.Timestamp("2015-06-22T06"):
                gdas1.t00z.pgrb2.0p25.f000...f003...f009
                gdas1.t00z.pgrb2.1p00.f000...f003...f009
            e.g.: s3://noaa-gfs-bdp-pds/gdas.20160101/00/gdas1.t00z.pgrb2.0p25.f000

        * t0 > pd.Timestamp("2016-05-11T06"):
                gdas1.t00z.pgrb2.0p25.f000...f001...f009
                gdas1.t00z.pgrb2.1p00.f000...f001...f009
            e.g.: s3://noaa-gfs-bdp-pds/gdas.20160601/00/gdas1.t00z.pgrb2.0p25.f000

        * t0 > pd.Timestamp("2017-07-19T06"):
                gdas.t00z.pgrb2.0p25.f000...f001...f009
                gdas.t00z.pgrb2.1p00.f000...f001...f009
            e.g.: s3://noaa-gfs-bdp-pds/gdas.20210101/00/gdas.t00z.pgrb2.0p25.f000

        * t0 > pd.Timestamp("2021-03-22T06")
                gdas.t00z.pgrb2.0p25.f000...f001...f009
                gdas.t00z.pgrb2.1p00.f000...f001...f009
            e.g.: s3://noaa-gfs-bdp-pds/gdas.20220101/00/atmos/gdas.t00z.pgrb2.0p25.f000

        * AWS at https://registry.opendata.aws/noaa-gfs-bdp-pds/
    """

    sample_dims = ("t0", "fhr")
    horizontal_dims = ("latitude", "longitude")
    file_suffixes = ("", "b")
    static_vars = ("lsm", "orog")

    @property
    def available_levels(self) -> tuple:
        return (
            1,
            2,
            3,
            5,
            7,
            10,
            15,
            20,
            30,
            40,
            50,
            70,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            275,
            300,
            325,
            350,
            375,
            400,
            425,
            450,
            475,
            500,
            525,
            550,
            575,
            600,
            625,
            650,
            675,
            700,
            725,
            750,
            775,
            800,
            825,
            850,
            875,
            900,
            925,
            950,
            975,
            1000,
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
        resolution: str,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
        accum_hrs: Optional[dict] = None,
    ) -> None:
        """
        Args:
            t0 (dict): Dictionary with start and end times for initial conditions, and e.g. "freq=6h". All options get passed to ``pandas.date_range``.
            fhr (dict): Dictionary with 'start', 'end', and 'step' forecast hours.
            resolution (str): spatial resolution, 0p25: 0.25 degree; 1p00: 1.0 degree
            variables (list, tuple, optional): variables to grab
            levels (list, tuple, optional): vertical levels to grab
            use_nearest_levels (bool, optional): if True, all level selection with
                ``xarray.Dataset.sel(level=levels, method="nearest")``
            slices (dict, optional): either "sel" or "isel", with slice, passed to xarray
            accum_hrs (dict, optional): period in hours over which to accumulate accumulated variables, e.g.
                {"accum_tp": 1}
                would be the same as passing filter_by_keys={"stepRange": "5-6"}
                when reading fhr 6 data using xarray and cfgrib
        """
        self.t0 = pd.date_range(**t0)
        self.fhr = np.arange(fhr["start"], fhr["end"] + 1, fhr["step"])
        self.resolution = resolution
        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
            accum_hrs=accum_hrs,
        )

        # make sure the default variable set "just works"
        # by pulling out the variable names only available at forecast time
        if tuple(self.variables) == self.available_variables:
            varlist = []
            for key, meta in self._varmeta.items():
                if not self._varmeta[key]["forecast_only"]:
                    varlist.append(key)
            self.variables = varlist

        # for GDAS, plenty of variables only exist in the forecast, not analysis grib files
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
        Build the file path to a GDAS GRIB file on AWS.
        Args:
            t0 (pd.Timestamp): The initial condition timestamp.
            fhr (int): The forecast hour.
            file_suffix (str): For GDAS,''
        Returns:
            str: The constructed file path.
        """
        res = self.resolution
        bucket = "s3://noaa-gfs-bdp-pds"
        outer = f"gdas.{t0.year:04d}{t0.month:02d}{t0.day:02d}/{t0.hour:02d}"
        if t0 > pd.Timestamp("2016-05-11T06"):
            fname = f"gdas1.t{t0.hour:02d}z.pgrb2.{res}.f{fhr:03d}"
            if t0 > pd.Timestamp("2017-07-19T06"):
                fname = f"gdas.t{t0.hour:02d}z.pgrb2.{res}.f{fhr:03d}"
                if t0 > pd.Timestamp("2021-03-22T06"):
                    fname = "atmos/" + fname
        else:
            msg = f"{self.name}.__init__: Data corresponding to the current date are not available on AWS."
            raise Exception(msg)

        fullpath = f"filecache::{bucket}/{outer}/{fname}"
        logger.debug(f"{self.name}._build_path: reading {fullpath}")
        return fullpath
