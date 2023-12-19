import yaml
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod

from .gaussian_grid import gaussian_latitudes

try:
    import xesmf as xe
except ImportError:
    raise ImportError(
        f"Cannot import xesmf. Install with 'conda install -c conda-forge xesmf',"
        f" and note the environment must be deactivated and reactivated"
    )


class RegridUFS(ABC):
    """
    Regrid a UFS dataset

    Example:

        Construct the output grid. The following static methods are provided in this class
            - compute_gaussian_grid
            - compute_latlon_grid
            - read_grid

        >>> lats, lons = RegridUFS.compute_gaussian_grid(180, 360)

        Open cice dataset using xarray or preferably :meth:`CICE6Dataset.open_dataset`

        >>> ds = xr.open_mfdataset("./input/{ocn_,iceh_}*.nc")

        Construct RegridCICE6 object, specifying output grid lats & lons, dataset and config file

        >>> rg = RegridUFS(lats, lons, ds, config_filename = "config-replay.yaml")

        Regrid dataset

        >>> ds = rg.regrid(ds)
    """

    def __init__(
        self,
        lats1d_out: np.array,
        lons1d_out: np.array,
        ds_in: xr.Dataset,
        config_filename: str,
        interp_method: str = "bilinear",
    ) -> None:
        """
        Initialize the RegridCICE6 object.
        Args:
            lats1d_out (array_like): One-dimensional array containing latitudes of the output grid.
            lons1d_out (array_like): One-dimensional array containing longitudes of the output grid.
            config_filename (str): YAML config file specifying options for regridding.
            ds_in (xarray.Dataset): CICE6 xarray Dataset
            interp_method (str, optional): Type of interpolation, default="bilinear".
        """
        super(RegridUFS, self).__init__()
        name = self.__class__.__name__

        # Load configuration from YAML file
        with open(config_filename, "r") as f:
            self.config = yaml.safe_load(f).get(name, {})

        # specify an output resolution
        self.lats1d_out = lats1d_out
        self.lons1d_out = lons1d_out
        self.interp_method = interp_method

        # create regridder
        self._create_regridder(ds_in)

    @abstractmethod
    def _create_regridder(self, ds_in: xr.Dataset) -> None:
        """
        Create regridder: computes weights and creates three xesmf.Regridder instances.
        Args:
            ds_in (xarray.Dataset): xarray dataset needed for creating the regridder.
        """

    @abstractmethod
    def regrid(self, ds_in: xr.Dataset) -> xr.Dataset:
        """
        Regrid an xarray dataset.
        Args:
            ds_in (xarray.Dataset): Input xarray Dataset to regrid.

        Returns:
            ds_out (xarray.Dataset): Regridded xarray Dataset
        """

    @staticmethod
    def compute_gaussian_grid(nlat, nlon) -> (np.array, np.array):
        """Compute gaussian grid latitudes and longitudes"""
        latitudes, _ = gaussian_latitudes(nlat // 2)
        longitudes = np.linspace(0, 360, nlon, endpoint=False)
        return latitudes, longitudes

    @staticmethod
    def compute_latlon_grid(nlat, nlon) -> (np.array, np.array):
        """Compute regular latlong grid coordinates"""
        latitudes = np.linspace(-90, 90, nlat, endpoint=False)
        longitudes = np.linspace(0, 360, nlon, endpoint=False)
        return latitudes, longitudes

    @staticmethod
    def read_grid(
        file, lats_name="grid_yt", lons_name="grid_xt"
    ) -> (np.array, np.array):
        """Read grid latitudes and longitudes from a netcdf file"""
        ds = xr.open_dataset(file)
        latitudes = ds[lats_name].values
        longitudes = ds[lons_name].values
        ds.close()
        return latitudes, longitudes
