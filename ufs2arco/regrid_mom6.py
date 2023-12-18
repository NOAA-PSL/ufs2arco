import os
import yaml
import warnings
import xarray as xr
import numpy as np

try:
    import xesmf as xe

    _has_xesmf = True
except ImportError:
    _has_xesmf = False

from .gaussian_grid import gaussian_latitudes


class RegridMOM6:
    """
    Regrid ocean dataset that is on a tripolar grid to a different grid (primarily Gaussian grid).


    Optional fields in config:
        rotation_file (str): path to file containing rotation fields "sin_rot" and "cos_rot", required for regridding vector fields
        weights_file_t2t (str): path to t2t interpolation weights file
        weights_file_u2t (str): path to u2t interpolation weights file
        weights_file_v2t (str): path to v2t interpolation weights file
        periodic (bool): Is the grid periodic in longitude?

    Example:

        Construct the output grid. The following static methods are provided in this class
            - compute_gaussian_grid
            - compute_latlon_grid
            - read_grid

        >>> lats, lons = RegridMOM6.compute_gaussian_grid(180, 360)

        Open ocean dataset using xarray or preferably :meth:`MOM6Dataset.open_dataset`

        >>> ds = xr.open_mfdataset("./input/ocn_1994_08_02_??.nc")

        Construct RegridMOM6 object, specifying output grid lats & lons, dataset and config file

        >>> rg = RegridMOM6(lats, lons, ds, config_filename = "config-replay.yaml")

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
        Initialize the RegridMOM6 object.
        Args:
            lats1d_out (array_like): One-dimensional array containing latitudes of the output grid.
            lons1d_out (array_like): One-dimensional array containing longitudes of the output grid.
            config_filename (str): YAML config file specifying options for regridding.
            ds_in (xarray.Dataset): MOM6 xarray Dataset
            interp_method (str, optional): Type of interpolation, default="bilinear".
        """
        if not _has_xesmf:
            raise ImportError(
                f"Cannot import xesmf. Install with 'conda install -c conda-forge xesmf',"
                f" and note the environment must be deactivated and reactivated"
            )

        super(RegridMOM6, self).__init__()
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

    def _create_regridder(self, ds_in: xr.Dataset) -> None:
        """
        Create regridder: computes weights and creates three xesmf.Regridder instances.
        Args:
            ds_in (xarray.Dataset): MOM6 xarray dataset needed for creating the regridder.

        """
        self.rotation_file = self.config.get("rotation_file", None)
        if self.rotation_file is None:
            warnings.warn(
                f"RegridMOM6.__init__: Could not find 'rotation_file' in configuration yaml. "
                f"Vector fields will be silently ignored."
            )

        # create renamed datasets
        ds_in_t = ds_in.rename({"xh": "lon", "yh": "lat", "z_l": "lev", "zl": "lev1"})
        ds_in_u = ds_in.rename({"xq": "lon", "yh": "lat", "z_l": "lev", "zl": "lev1"})
        ds_in_v = ds_in.rename({"xh": "lon", "yq": "lat", "z_l": "lev", "zl": "lev1"})

        # open rotation dataset
        if self.rotation_file is not None:
            ds_rot = xr.open_dataset(self.rotation_file)
            ds_rot = ds_rot[["cos_rot", "sin_rot"]]
            self.ds_rot = ds_rot.rename({"xh": "lon", "yh": "lat"})

        # create output dataset
        lons, lats = np.meshgrid(self.lons1d_out, self.lats1d_out)
        grid_out = xr.Dataset()
        grid_out["lon"] = xr.DataArray(lons, dims=["lat", "lon"])
        grid_out["lat"] = xr.DataArray(lats, dims=["lat", "lon"])

        # get nlon/nlat for input/output datsets
        nlon_i = ds_in.sizes["yh"]
        nlat_i = ds_in.sizes["xh"]
        nlon_o = len(self.lons1d_out)
        nlat_o = len(self.lats1d_out)
        self.ires = f"{nlon_i}x{nlat_i}"
        self.ores = f"{nlon_o}x{nlat_o}"

        # paths to interpolation weights files
        if "weights_file_t2t" in self.config.keys():
            weights_file_t2t = self.config["weights_file_t2t"]
            weights_file_u2t = self.config["weights_file_u2t"]
            weights_file_v2t = self.config["weights_file_v2t"]
        if weights_file_t2t is None:
            weights_file_t2t = (
                f"weights-{self.ires}.Ct.{self.ores}.Ct.{self.interp_method}.nc"
            )
            weights_file_u2t = (
                f"weights-{self.ires}.Cu.{self.ires}.Ct.{self.interp_method}.nc"
            )
            weights_file_v2t = (
                f"weights-{self.ires}.Cv.{self.ires}.Ct.{self.interp_method}.nc"
            )

        # create regridding instances
        periodic = self.config["periodic"]
        reuse = os.path.exists(weights_file_t2t)
        self.rg_tt = xe.Regridder(
            ds_in_t,
            grid_out,
            self.interp_method,
            periodic=periodic,
            reuse_weights=reuse,
            filename=weights_file_t2t,
        )
        if self.rotation_file is not None:
            reuse = os.path.exists(weights_file_u2t)
            self.rg_ut = xe.Regridder(
                ds_in_u,
                ds_in_t,
                self.interp_method,
                periodic=periodic,
                reuse_weights=reuse,
                filename=weights_file_u2t,
            )
            reuse = os.path.exists(weights_file_v2t)
            self.rg_vt = xe.Regridder(
                ds_in_v,
                ds_in_t,
                self.interp_method,
                periodic=periodic,
                reuse_weights=reuse,
                filename=weights_file_v2t,
            )

    def regrid(self, ds_in: xr.Dataset) -> xr.Dataset:
        """
        Regrid a MOM6 xarray dataset.
        Args:
            ds_in (xarray.Dataset): Input MOM6 xarray Dataset to regrid.

        Returns:
            ds_out (xarray.Dataset): Regridded MOM6 xarray Dataset
        """
        ds_out = []

        # MOM6 dataset specific variable and coordinate names
        coords_xy = {"yh", "xh", "yq", "xq"}
        variable_map = {
            "SSU": ("SSV", "U"),
            "SSV": (None, "skip"),
            "uo": ("vo", "U"),
            "vo": (None, "skip"),
            "taux": ("tauy", "U"),
            "tauy": (None, "skip"),
        }
        ds_in_t = ds_in.rename({"xh": "lon", "yh": "lat", "z_l": "lev", "zl": "lev1"})
        ds_in_u = ds_in.rename({"xq": "lon", "yh": "lat", "z_l": "lev", "zl": "lev1"})
        ds_in_v = ds_in.rename({"xh": "lon", "yq": "lat", "z_l": "lev", "zl": "lev1"})

        # interate through variables and regrid
        for var in list(ds_in.keys()):
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

            # helper to update attrs of da_out and append to ds_out
            def update_attrs(da_out, var):
                da_out.attrs.update(
                    {
                        "long_name": ds_in[var].long_name,
                        "units": ds_in[var].units,
                    }
                )
                ds_out.append(da_out.to_dataset(name=var))

            # scalar fields
            if pos == "T":
                # interpolate
                da_out = self.rg_tt(ds_in_t[var])
                update_attrs(da_out, var)

            # vector fields
            if pos == "U" and self.rotation_file is not None:
                # interpolate to t-points
                interp_u = self.rg_ut(ds_in_u[var])
                interp_v = self.rg_vt(ds_in_v[var2])

                # rotate to earth-relative
                urot = interp_u * self.ds_rot.cos_rot + interp_v * self.ds_rot.sin_rot
                vrot = interp_v * self.ds_rot.cos_rot - interp_u * self.ds_rot.sin_rot

                # interoplate
                uinterp_out = self.rg_tt(urot)
                vinterp_out = self.rg_tt(vrot)

                # append vars
                update_attrs(uinterp_out, var)
                update_attrs(vinterp_out, var2)

        # merge dataarrays into a dataset and set attributes
        ds_out = xr.merge(ds_out, compat="override")
        ds_out.attrs = ds_in.attrs
        ds_out = ds_out.assign_coords(lon=("lon", self.lons1d_out))
        ds_out = ds_out.assign_coords(lat=("lat", self.lats1d_out))
        ds_out = ds_out.assign_coords(lev=("lev", ds_in.z_l.values))
        ds_out = ds_out.assign_coords(lev1=("lev1", ds_in.zl.values))
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
        ds_out["lev"].attrs.update(
            {
                "units": "meters",
                "axis": "Z",
                "positive": "down",
            }
        )
        ds_out["lev1"].attrs.update(
            {
                "units": "meters",
                "axis": "Z",
                "positive": "down",
            }
        )

        return ds_out
