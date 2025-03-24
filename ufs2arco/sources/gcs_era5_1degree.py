

class GCSERA5OneDegree(Source):

    static_vars = ("land_sea_mask", "geopotential_at_surface")
    sample_dims = ("t0", "fhr", "member")
    base_dims = ("level", "latitude", "longitude")

    _is_open = False

    @property
    def rename(self) -> dict:
        return {
            "time": "t0",
        }

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

        # rename
        for key, val in self.rename.items():
            if key in xds:
                xds = xds.rename({key: val})

        # select
        xds = xds[self.variables]
        xds = xds.sel(level=self.levels)

        # set
        self._xds = xds
        self._is_open = True

    def open_sample_dataset(
        self,
        t0: pd.Timestamp,
        fhr: int,
        member: int,
        cache_dir: str,
        open_static_vars: bool,
    ) -> xr.Dataset:

        if not self._is_open:
            self._open_dataset()

        xds = self._xds.sel(t0=[t0])

        # do this stuff to play nicely with the rest of the repo
        xds = xds.expand_dims({"fhr": [0], "member": [0]})

        selection = list(self.variables) if open_static_vars else list(self.dynamic_vars)
        return xds[selection]
