import os
import yaml
import warnings
import xarray as xr
import numpy as np

from .ufsregridder import UFSRegridder

try:
    import xesmf as xe
except ImportError:
    pass


class CICE6Regridder(UFSRegridder):
    """
    Regrid cice dataset that is on a tripolar grid to a different grid (primarily Gaussian grid).


    Optional fields in config:
        rotation_file (str): path to file containing rotation fields "sin_rot" and "cos_rot", required for regridding vector fields
        weights_file_t2t (str): path to t2t interpolation weights file
        weights_file_u2t (str): path to u2t interpolation weights file
        periodic (bool): Is the grid periodic in longitude?
    """

    __doc__ = __doc__ + UFSRegridder.__doc__

    rg_tt = None
    rg_ut = None

    def __init__(
        self,
        lats1d_out: np.array,
        lons1d_out: np.array,
        ds_in: xr.Dataset,
        config_filename: str,
    ) -> None:
        super(CICE6Regridder, self).__init__(
            lats1d_out, lons1d_out, ds_in, config_filename
        )

    def _create_regridder(self, ds_in: xr.Dataset) -> None:

        # create rotation dataset
        self.rotation_file = self.config.get("rotation_file", None)
        self.ds_rot = None
        if self.rotation_file is not None:
            ds_rot = xr.open_dataset(self.rotation_file)
            self.ds_rot = xr.Dataset()
            self.ds_rot["cos_rot"] = np.cos(ds_rot["ANGLE"])
            self.ds_rot["sin_rot"] = np.sin(ds_rot["ANGLE"])
        elif "ANGLE" in ds_in:
            self.ds_rot = xr.Dataset()
            self.ds_rot["cos_rot"] = np.cos(ds_in["ANGLE"])
            self.ds_rot["sin_rot"] = np.sin(ds_in["ANGLE"])
        else:
            warnings.warn(
                f"CICE6Regridder._create_regridder: Could not find 'rotation_file' in configuration yaml. "
                f"Vector fields will be silently ignored."
            )

        # create input dataset with t-/u- coordinates
        ds_in_t = xr.Dataset()
        ds_in_t["lon"] = ds_in["TLON"]
        ds_in_t["lat"] = ds_in["TLAT"]
        ds_in_u = xr.Dataset()
        ds_in_u["lon"] = ds_in["ULON"]
        ds_in_u["lat"] = ds_in["ULAT"]

        # create output dataset
        lons, lats = np.meshgrid(self.lons1d_out, self.lats1d_out)
        grid_out = xr.Dataset()
        grid_out["lon"] = xr.DataArray(lons, dims=["lat", "lon"])
        grid_out["lat"] = xr.DataArray(lats, dims=["lat", "lon"])

        # get nlon/nlat for input/output datsets
        nlon_i = ds_in.sizes["ni"]
        nlat_i = ds_in.sizes["nj"]
        nlon_o = len(self.lons1d_out)
        nlat_o = len(self.lats1d_out)
        self.ires = f"{nlon_i}x{nlat_i}"
        self.ores = f"{nlon_o}x{nlat_o}"

        # paths to interpolation weights files
        wfiles = dict()
        interp_method = self.config["interp_method"]
        for key in ["weights_file_t2t", "weights_file_u2t"]:
            vin = key[-3]
            default = f"weights-cice6-{self.ires}.C{vin}.{self.ores}.Ct.{interp_method}.nc"
            path = self.config.get(key, None)
            wfiles[key] = path if path is not None else default

        # create regridding instances
        periodic = self.config["periodic"]
        reuse = os.path.exists(wfiles["weights_file_t2t"])
        self.rg_tt = xe.Regridder(
            ds_in_t,
            grid_out,
            interp_method,
            periodic=periodic,
            reuse_weights=reuse,
            filename=wfiles["weights_file_t2t"],
        )
        if self.ds_rot is not None:
            reuse = os.path.exists(wfiles["weights_file_u2t"])
            self.rg_ut = xe.Regridder(
                ds_in_u,
                ds_in_t,
                interp_method,
                periodic=periodic,
                reuse_weights=reuse,
                filename=wfiles["weights_file_u2t"],
            )

    def regrid(self, ds_in: xr.Dataset) -> xr.Dataset:
        ds_out = []

        # CICE6 dataset specific variable and coordinate names
        coords_xy = {"nj", "ni"}
        variable_map = {
            "uvel": ("vvel", "U"),
            "vvel": (None, "skip"),
            "strairx": ("strairy", "U"),
            "strairy": (None, "skip"),
            "strocnx": ("strocny", "U"),
            "strocny": (None, "skip"),
            "uvel_d": ("vvel_d", "U"),
            "vvel_d": (None, "skip"),
            "strairx_d": ("strairy_d", "U"),
            "strairy_d": (None, "skip"),
            "strocnx_d": ("strocny_d", "U"),
            "strocny_d": (None, "skip"),
            "uvel_h": ("vvel_h", "U"),
            "vvel_h": (None, "skip"),
            "strairx_h": ("strairy_h", "U"),
            "strairy_h": (None, "skip"),
            "strocnx_h": ("strocny_h", "U"),
            "strocny_h": (None, "skip"),
        }

        return super(CICE6Regridder, self).regrid_tripolar(
            ds_in,
            self.ds_rot,
            self.rg_tt,
            self.rg_ut,
            self.rg_ut,
            coords_xy,
            variable_map,
        )
