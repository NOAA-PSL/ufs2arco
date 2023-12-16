import os
import yaml
import xarray as xr
import numpy as np
import xesmf as xe
from gaussian_grid import gaussian_latitudes


class RegridOcean:
    """
    Regrid ocean dataset that is on a tripolar grid to a different grid (primarily Gaussian grid).

    Required fields in config:
        input_path (str): path where ocean netcdf files (ocn_*.nc) are located
        output_path (str): path where regridded data and interpolation weights are stored
        rotation_file (str): path to file containing rotation fields "sin_rot" and "cos_rot"

    Optional fields in config:
        weights_file_t2t (str): path to t2t interpolation weights file
        weights_file_u2t (str): path to u2t interpolation weights file
        weights_file_v2t (str): path to v2t interpolation weights file

    Example:

        Construct the output grid. The following static methods are provided in this class
            - compute_gaussian_grid
            - compute_latlon_grid
            - read_grid

        >>> lats, lons = RegridOcean.compute_gaussian_grid(180, 360)

        Construct RegridOcean object, specifying output grid lats & lons and a config file

        >>> rg = RegridOcean(lats, lons, config_filename = "config-regrid-ocean.yaml")

        Compile list of files to regrid
        >>> files = glob.glob(f"./input/ocn_2016_08_02_??.nc")
        >>> files.sort()

        Create regridder to compute interpolation weights
        >>> rg.create_regridder(files[0])

        Regrid all files using weights computed above
        >>> for file in files:
        >>>    rg.regrid(file)
    """

    def __init__(
        self,
        lats1d_out: np.array,
        lons1d_out: np.array,
        config_filename: str,
        interp_method: str = "bilinear",
    ) -> None:
        """
        Initialize the RegridOcean object.
        :param lats1d_out: One-dimensional array containing latitudes of the output grid.
        :param lons1d_out: One-dimensional array containing longitudes of the output grid.
        :param config_filename: YAML config file specifying options for regridding.
        :param interp_method: Type of interpolation, default="bilinear".
        """

        super(RegridOcean, self).__init__()
        name = self.__class__.__name__

        # Load configuration from YAML file
        with open(config_filename, "r") as f:
            self.config = yaml.safe_load(f).get(name, {})

        # specify an output resolution
        self.lats1d_out = lats1d_out
        self.lons1d_out = lons1d_out
        self.interp_method = interp_method

    @staticmethod
    def compute_gaussian_grid(nlat, nlon):
        """Compute gaussian grid latitudes and longitudes"""
        latitudes, _ = gaussian_latitudes(nlat // 2)
        longitudes = np.linspace(0, 360, nlon, endpoint=False)
        return latitudes, longitudes

    @staticmethod
    def compute_latlon_grid(nlat, nlon):
        """Compute regular latlong grid coordinates"""
        latitudes = np.linspace(-90, 90, nlat, endpoint=False)
        longitudes = np.linspace(0, 360, nlon, endpoint=False)
        return latitudes, longitudes

    @staticmethod
    def read_grid(file, lats_name="grid_yt", lons_name="grid_xt"):
        """Read grid latitudes and longitudes from a netcdf file"""
        ds = xr.open_dataset(file)
        latitudes = ds[lats_name].values
        longitudes = ds[lons_name].values
        ds.close()
        return latitudes, longitudes

    def create_regridder(self, file):
        """
        Create regridder: computes weights and creates three xesmf.Regridder instances.
        :param file: Ocean input file for creating the regridder.
        """
        self.input_path = self.config["input_path"]
        self.rotation_file = self.config["rotation_file"]
        self.output_path = self.config["output_path"]

        # open input dataset
        ds_in = xr.open_dataset(file)
        ds_in_t = ds_in.rename({"xh": "lon", "yh": "lat"})
        ds_in_u = ds_in.rename({"xq": "lon", "yh": "lat"})
        ds_in_v = ds_in.rename({"xh": "lon", "yq": "lat"})

        # open rotation dataset
        file_rot = f"{self.rotation_file}"
        ds_rot = xr.open_dataset(file_rot)
        ds_rot = ds_rot[["cos_rot", "sin_rot"]]
        self.ds_rot = ds_rot.rename({"xh": "lon", "yh": "lat"})

        # create output dataset
        lons, lats = np.meshgrid(self.lons1d_out, self.lats1d_out)
        grid_out = xr.Dataset()
        grid_out["lon"] = xr.DataArray(lons, dims=["nx", "ny"])
        grid_out["lat"] = xr.DataArray(lats, dims=["nx", "ny"])

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
        else:
            weights_file_t2t = f"{self.output_path}/weights-{self.ires}.Ct.{self.ores}.Ct.{self.interp_method}.nc"
            weights_file_u2t = f"{self.output_path}/weights-{self.ires}.Cu.{self.ires}.Ct.{self.interp_method}.nc"
            weights_file_v2t = f"{self.output_path}/weights-{self.ires}.Cv.{self.ires}.Ct.{self.interp_method}.nc"

        # create regridding instances
        reuse = os.path.exists(weights_file_t2t)
        self.rg_tt = xe.Regridder(
            ds_in_t,
            grid_out,
            self.interp_method,
            periodic=True,
            reuse_weights=reuse,
            filename=weights_file_t2t,
        )
        reuse = os.path.exists(weights_file_u2t)
        self.rg_ut = xe.Regridder(
            ds_in_u,
            ds_in_t,
            self.interp_method,
            periodic=True,
            reuse_weights=reuse,
            filename=weights_file_u2t,
        )
        reuse = os.path.exists(weights_file_v2t)
        self.rg_vt = xe.Regridder(
            ds_in_v,
            ds_in_t,
            self.interp_method,
            periodic=True,
            reuse_weights=reuse,
            filename=weights_file_v2t,
        )

    def regrid(self, file):
        """
        Regrid a single ocean file.
        :param file: Path to input file to regrid.
        """
        ds_in = xr.open_dataset(file)
        ds_out = []

        for var in list(ds_in.keys()):
            if len(ds_in[var].coords) > 2:
                coords = ds_in[var].coords.to_index()

                # choose regridding type
                if coords.names[0] != "time":
                    raise ValueError("First coordinate should be time")
                else:
                    variable_map = {
                        "SSU": ("SSV", "U"),
                        "SSV": (None, "skip"),
                        "uo": ("vo", "U"),
                        "vo": (None, "skip"),
                        "taux": ("tauy", "U"),
                        "tauy": (None, "skip"),
                    }
                    if var in variable_map.keys():
                        var2, pos = variable_map[var]
                    else:
                        var2, pos = (None, "T")

                # 3-dimensional data?
                if coords.names[1] == "z_l":
                    dims = ["time", "lev", "lat", "lon"]
                elif coords.names[1] == "zl":
                    # Should this be interpolated to "z_l" instead
                    dims = ["time", "lev1", "lat", "lon"]
                else:
                    dims = ["time", "lat", "lon"]

                # scalar fields
                if pos == "T":
                    da_out = self.rg_tt(ds_in[var])
                    da_out = da_out.rename(dict(zip(da_out.dims, dims)))
                    da_out.attrs.update(
                        {
                            "long_name": ds_in[var].long_name,
                            "units": ds_in[var].units,
                        }
                    )
                    ds_out.append(da_out.to_dataset(name=var))

                # vector fields
                if pos == "U":
                    # interplate u and v to t-point, then rotate currents/winds to
                    # earth relative before interpolation

                    # interpolate to t-points
                    interp_u = self.rg_ut(ds_in[var])
                    interp_v = self.rg_ut(ds_in[var2])

                    # rotate to earth-relative
                    urot = (
                        interp_u * self.ds_rot.cos_rot + interp_v * self.ds_rot.sin_rot
                    )
                    vrot = (
                        interp_v * self.ds_rot.cos_rot - interp_u * self.ds_rot.sin_rot
                    )

                    # interoplate
                    uinterp_out = self.rg_tt(urot)
                    vinterp_out = self.rg_tt(vrot)

                    # construct u datarray
                    da_out = uinterp_out
                    da_out = da_out.rename(dict(zip(da_out.dims, dims)))
                    da_out.attrs.update(
                        {
                            "long_name": ds_in[var].long_name,
                            "units": ds_in[var].units,
                        }
                    )
                    ds_out.append(da_out.to_dataset(name=var))

                    # construct v datarray
                    da_out = vinterp_out
                    da_out = da_out.rename(dict(zip(da_out.dims, dims)))
                    da_out.attrs.update(
                        {
                            "long_name": ds_in[var2].long_name,
                            "units": ds_in[var2].units,
                        }
                    )
                    ds_out.append(da_out.to_dataset(name=var2))

        # merge dataarrays into a dataset and set attributes
        ds_out = xr.merge(ds_out)
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

        # insert resolution ot output file name
        basename = os.path.basename(file)
        output_file = f"{self.output_path}/{basename[:-3]}_{self.ores}{basename[-3:]}"
        ds_out.to_netcdf(output_file)

        # close datasets
        ds_out.close()
        ds_in.close()
