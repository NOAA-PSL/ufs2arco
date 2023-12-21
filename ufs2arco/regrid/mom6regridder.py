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


class MOM6Regridder(UFSRegridder):
    """
    Regrid ocean dataset that is on a tripolar grid to a different grid (primarily Gaussian grid).


    Optional fields in config:
        rotation_file (str): path to file containing rotation fields "sin_rot" and "cos_rot", required for regridding vector fields
        weights_file_t2t (str): path to t2t interpolation weights file
        weights_file_u2t (str): path to u2t interpolation weights file
        weights_file_v2t (str): path to v2t interpolation weights file
        periodic (bool): Is the grid periodic in longitude?
    """

    rg_tt = None
    rg_ut = None
    rg_vt = None

    __doc__ = __doc__ + UFSRegridder.__doc__

    def __init__(
        self,
        lats1d_out: np.array,
        lons1d_out: np.array,
        ds_in: xr.Dataset,
        config_filename: str,
        interp_method: str = "bilinear",
    ) -> None:
        super(MOM6Regridder, self).__init__(
            lats1d_out, lons1d_out, ds_in, config_filename, interp_method
        )

    def _create_regridder(self, ds_in: xr.Dataset) -> None:

        # create rotation dataset
        self.rotation_file = self.config.get("rotation_file", None)
        self.ds_rot = None
        if self.rotation_file is not None:
            ds_rot = xr.open_dataset(self.rotation_file)
            ds_rot = ds_rot[["cos_rot", "sin_rot"]]
            self.ds_rot = ds_rot.rename({"xh": "lon", "yh": "lat"})
        elif "cos_rot" in ds_in and "sin_rot" in ds_in:
            self.ds_rot = xr.Dataset()
            self.ds_rot["cos_rot"] = ds_in["cos_rot"]
            self.ds_rot["sin_rot"] = ds_in["sin_rot"]
        else:
            warnings.warn(
                f"MOM6Regridder._create_regridder: Could not find 'rotation_file' in configuration yaml. "
                f"Vector fields will be silently ignored."
            )

        # create renamed datasets
        ds_in_t = ds_in.rename({"xh": "lon", "yh": "lat"})
        ds_in_u = ds_in.rename({"xq": "lon", "yh": "lat"})
        ds_in_v = ds_in.rename({"xh": "lon", "yq": "lat"})

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
        wfiles = dict()
        for key in ["weights_file_t2t", "weights_file_u2t", "weights_file_v2t"]:
            vin = key[-3]
            default = f"weights-mom6-{self.ires}.C{vin}.{self.ores}.Ct.{self.interp_method}.nc"
            path = self.config.get(key, None)
            wfiles[key] = path if path is not None else default

        # create regridding instances
        periodic = self.config["periodic"]
        reuse = os.path.exists(wfiles["weights_file_t2t"])
        self.rg_tt = xe.Regridder(
            ds_in_t,
            grid_out,
            self.interp_method,
            periodic=periodic,
            reuse_weights=reuse,
            filename=wfiles["weights_file_t2t"],
        )
        if self.ds_rot is not None:
            reuse = os.path.exists(wfiles["weights_file_u2t"])
            self.rg_ut = xe.Regridder(
                ds_in_u,
                ds_in_t,
                self.interp_method,
                periodic=periodic,
                reuse_weights=reuse,
                filename=wfiles["weights_file_u2t"],
            )
            reuse = os.path.exists(wfiles["weights_file_v2t"])
            self.rg_vt = xe.Regridder(
                ds_in_v,
                ds_in_t,
                self.interp_method,
                periodic=periodic,
                reuse_weights=reuse,
                filename=wfiles["weights_file_v2t"],
            )

    def regrid(self, ds_in: xr.Dataset) -> xr.Dataset:

        # define MOM6 dataset specific coordinates and vector fields map
        coords_xy = {"yh", "xh", "yq", "xq"}
        variable_map = {
            "SSU": ("SSV", "U"),
            "SSV": (None, "skip"),
            "uo": ("vo", "U"),
            "vo": (None, "skip"),
            "taux": ("tauy", "U"),
            "tauy": (None, "skip"),
        }

        return super(MOM6Regridder, self).regrid_tripolar(
            ds_in,
            self.ds_rot,
            self.rg_tt,
            self.rg_ut,
            self.rg_vt,
            coords_xy,
            variable_map,
        )
