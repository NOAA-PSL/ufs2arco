import yaml
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
import warnings

from .gaussian_grid import gaussian_latitudes

try:
    from xesmf import Regridder

    _has_xesmf = True
except ImportError:
    Regridder = None
    _has_xesmf = False


class UFSRegridder(ABC):
    """

    Args:
        lats1d_out (array_like): One-dimensional array containing latitudes of the output grid.
        lons1d_out (array_like): One-dimensional array containing longitudes of the output grid.
        config_filename (str): YAML config file specifying options for regridding.
        ds_in (xarray.Dataset): CICE6 xarray Dataset
        interp_method (str, optional): Type of interpolation, default="bilinear".

    Example:

        This example uses the :class:`CICE6Regridder`, but similar capabilities exist for :class:`MOM6Regridder`.
        Construct the output grid. The following static methods are provided in this class
            - compute_gaussian_grid
            - compute_latlon_grid
            - read_grid

        >>> lats, lons = CICE6Regridder.compute_gaussian_grid(180, 360)

        Open dataset using xarray or preferably :meth:`CICE6Dataset.open_dataset`

        >>> ds = xr.open_mfdataset("./input/{ocn_,iceh_}*.nc")

        Construct :class:`CICE6Regridder` object, specifying output grid lats & lons, dataset and config file

        >>> rg = CICE6Regridder(lats, lons, ds, config_filename = "config-replay.yaml")

        Regrid dataset

        >>> ds = rg.regrid(ds)
    """

    def __init__(
        self,
        lats1d_out: np.array,
        lons1d_out: np.array,
        ds_in: xr.Dataset,
        config_filename: str,
    ) -> None:
        """
        Initialize the UFSRegridder object.
        """

        if not _has_xesmf:
            raise ImportError(
                f"Cannot import xesmf. Install with 'conda install -c conda-forge xesmf',"
                f" and note the environment must be deactivated and reactivated"
            )

        super(UFSRegridder, self).__init__()
        name = self.__class__.__name__
        # Look for the "regrid" section of e.g. MOM6Dataset
        name = name.replace("Regridder", "Dataset")

        # Load configuration from YAML file
        with open(config_filename, "r") as f:
            dataset_config = yaml.safe_load(f).get(name, {})
            self.config = dataset_config.get("regrid", {})
            self.data_vars = dataset_config.get("data_vars", "")

        # specify an output resolution
        self.lats1d_out = lats1d_out
        self.lons1d_out = lons1d_out
        self.interp_method = self.config["interp_method"]

        # create regridder
        self._create_regridder(ds_in)

    @abstractmethod
    def _create_regridder(self, ds_in: xr.Dataset) -> None:
        """
        Create regridder: computes weights and creates three xesmf.Regridder instances.
        Args:
            ds_in (xarray.Dataset): xarray dataset needed for creating the regridder.
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

    def regrid_tripolar(
        self,
        ds_in: xr.Dataset,
        ds_rot: xr.Dataset,
        rg_tt: Regridder,
        rg_ut: Regridder,
        rg_vt: Regridder,
        coords_xy: set,
        variable_map: dict,
    ) -> xr.Dataset:
        """
        Regrid an xarray dataset.

        Args:
            ds_in (xarray.Dataset): Input xarray Dataset to regrid.
            ds_rot (xarray.Dataset): Dataset containing rotation angles
            rg_tt (xesmf.Regridder): Regridder object mapping t-points to output grid
            rg_ut (xesmf.Regridder): Regridder object mapping u-points to t-points
            rg_vt (xesmf.Regridder): Regridder object mapping v-points to t-points
            coords_xy (set): A list of spatial coordinate names used in ds_in
            variable_map (dict): A dictionary mapping vector field names to a set e.g. sea-surface velocity
                       "SSU": ("SSV", "U"),
                       "SSV": (None, "skip)

        Returns:
            ds_out (xarray.Dataset): Regridded xarray Dataset
        """
        ds_out = []

        # Only loop through data variables we want to keep
        # If user didn't give any, then loop through all variables
        data_vars = self.data_vars if len(self.data_vars) > 0 else list(ds_in.keys())

        # make sure desired variables are actually in the dataset
        data_vars = [k for k in data_vars if k in ds_in]

        # interate through variables and regrid
        for var in data_vars:
            coords = ds_in[var].coords.to_index()

            # must have lat/lon like coordinates, otherwise append copy
            if len(set(coords.names) & coords_xy) < 2:
                ds_out.append(ds_in[var])
                continue

            # choose regridding type
            if var in variable_map.keys():
                var2, pos = variable_map[var]
            else:
                var2, pos = (None, "T")

            # scalar fields
            if pos == "T":
                # interpolate
                da_out = rg_tt(ds_in[var])
                ds_out.append(_xda_to_xds(da_out, var, ds_in[var].attrs))

            # vector fields
            if pos == "U" and ds_rot is not None:
                # interpolate to t-points
                interp_u = rg_ut(ds_in[var])
                interp_v = rg_vt(ds_in[var2])

                # rotate to earth-relative
                urot = (
                    interp_u * ds_rot.cos_rot + interp_v * ds_rot.sin_rot
                )
                vrot = (
                    interp_v * ds_rot.cos_rot - interp_u * ds_rot.sin_rot
                )

                # interoplate
                uinterp_out = rg_tt(urot)
                vinterp_out = rg_tt(vrot)

                # append vars
                ds_out.append(_xda_to_xds(uinterp_out, var, ds_in[var].attrs))
                ds_out.append(_xda_to_xds(vinterp_out, var2, ds_in[var2].attrs))

            elif pos == "U" and ds_rot is None:
                warnings.warn(
                    f"UFSRegridder.regrid_tripolar: rotation_file not provided, skipping vector fields ({var}, {var2})"
                )

        # merge dataarrays into a dataset and set attributes
        ds_out = xr.merge(ds_out, compat="override")
        ds_out.attrs = ds_in.attrs
        ds_out = ds_out.assign_coords(lon=("lon", self.lons1d_out))
        ds_out = ds_out.assign_coords(lat=("lat", self.lats1d_out))
        ds_out = ds_out.assign_coords(time=("time", ds_in.time.values))
        ds_out["lon"].attrs.update(
            {
                "units": "degrees_east",
                "axis": "X",
                "standard_name": "longitude",
            }
        )
        ds_out["lat"].attrs.update(
            {
                "units": "degrees_north",
                "axis": "Y",
                "standard_name": "latiitude",
            }
        )

        return ds_out


def _xda_to_xds(da_out, var, attrs):
    """helper to update attrs and prepare for ds_out"""
    xds = da_out.to_dataset(name=var)
    xds[var].attrs.update(
        {
            "long_name": attrs["long_name"],
            "units": attrs["units"],
        }
    )
    return xds
