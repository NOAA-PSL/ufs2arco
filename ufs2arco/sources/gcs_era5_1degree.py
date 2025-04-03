import logging
from typing import Optional

import pandas as pd
import xarray as xr

from ufs2arco.sources import AnalysisSource

logger = logging.getLogger("ufs2arco")


class GCSERA5OneDegree(AnalysisSource):

    static_vars = ("land_sea_mask", "geopotential_at_surface")

    _is_open = False

    @property
    def available_variables(self) -> tuple:
        return (
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "angle_of_sub_gridscale_orography",
            "anisotropy_of_sub_gridscale_orography",
            "geopotential",
            "geopotential_at_surface",
            "high_vegetation_cover",
            "lake_cover",
            "lake_depth",
            "land_sea_mask",
            "low_vegetation_cover",
            "mean_sea_level_pressure",
            "sea_ice_cover",
            "sea_surface_temperature",
            "slope_of_sub_gridscale_orography",
            "soil_type",
            "specific_humidity",
            "standard_deviation_of_filtered_subgrid_orography",
            "standard_deviation_of_orography",
            "surface_pressure",
            "temperature",
            "toa_incident_solar_radiation",
            "total_cloud_cover",
            "total_column_water_vapour",
            "total_precipitation",
            "type_of_high_vegetation",
            "type_of_low_vegetation",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
        )

    @property
    def available_levels(self) -> tuple:
        return (
            1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
            100, 125, 150, 175, 200, 225, 250,
            300, 350, 400, 450, 500, 550,
            600, 650, 700, 750, 775,
            800, 825, 850, 875,
            900, 925, 950, 975, 1000,
        )

    def _open_dataset(self):
        """May as well just do all of this once
        """

        # open
        xds = xr.open_zarr(
            "gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr",
            storage_options={"token": "anon"},
        )

        # select
        xds = xds[self.variables]
        if self.levels is not None:
            xds = xds.sel(level=self.levels, **self._level_sel_kwargs)
        xds = self.apply_slices(xds)

        # set
        self._xds = xds
        self._is_open = True

    def open_sample_dataset(
        self,
        time: pd.Timestamp,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:

        if not self._is_open:
            self._open_dataset()

        xds = self._xds.sel(time=[time])
        osv = open_static_vars or self._open_static_vars(time)
        selection = list(self.variables) if osv else list(self.dynamic_vars)

        xds = xds[selection].copy().load()
        return xds
