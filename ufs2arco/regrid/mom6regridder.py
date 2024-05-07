import os
import yaml
import warnings
import xarray as xr
import numpy as np
import cf_xarray as cfxr

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
    ) -> None:
        super(MOM6Regridder, self).__init__(
            lats1d_out, lons1d_out, ds_in, config_filename
        )

    def create_grid_in(
        self,
        mom6_grid: xr.Dataset,
    ) -> xr.Dataset:
        """Convert mom6 grid into a grid that is ready to be used by regridder.
        This comes from here: https://mom6-analysiscookbook.readthedocs.io/en/latest/notebooks/Horizontal_Remapping.html

        Args:
            mom6_grid (xr.Dataset):
                Dataset with all necessary mom6 metadata.

        Returns:
            ds_in_t (xr.Dataset):
                Dataset ready to be used as input grid for tracers.
            ds_in_v (xr.Dataset):
                Dataset ready to be used as input grid for v velocity.
            ds_in_u (xr.Dataset):
                Dataset ready to be used as input grid for u velocity.
        """
        grid_in = xr.Dataset()
        grid_in["lon"] = mom6_grid["geolon"]
        grid_in["lat"] = mom6_grid["geolat"]
        grid_in["lon_u"] = mom6_grid["geolon_u"]
        grid_in["lat_u"] = mom6_grid["geolat_u"]
        grid_in["lon_v"] = mom6_grid["geolon_v"]
        grid_in["lat_v"] = mom6_grid["geolat_v"]
        grid_in['cos_rot'] = mom6_grid["cos_rot"]
        grid_in['sin_rot'] = mom6_grid["sin_rot"]
        ny, nx = grid_in["lon"].shape
        lon_b = np.empty((ny + 1, nx + 1))
        lat_b = np.empty((ny + 1, nx + 1))
        lon_b[1:, 1:] = mom6_grid["geolon_c"].values
        lat_b[1:, 1:] = mom6_grid["geolat_c"].values
        # periodicity
        lon_b[:, 0] = lon_b[:, -1]
        lat_b[:, 0] = lat_b[:, -1]
        # south edge
        dy = (lat_b[2, :] - lat_b[1, :]).mean()
        lat_b[0, 1:] = lat_b[1, 1:] - dy
        lon_b[0, 1:] = lon_b[1, 1:]
        # corner point
        lon_b[0, 0] = lon_b[1, 0]
        lat_b[0, 0] = lat_b[0, 1]
        grid_in["lon_b"] = xr.DataArray(data=lon_b)
        grid_in["lat_b"] = xr.DataArray(data=lat_b)

        # create renamed datasets
        ds_in_t = grid_in[["lon", "lat", "lat_b", "lon_b"]]
        ds_in_u = grid_in[["lon_u", "lat_u", "lat_b", "lon_b"]].rename(
            {"lat_u": "lat", "lon_u": "lon"}
        )
        ds_in_v = grid_in[["lon_v", "lat_v", "lat_b", "lon_b"]].rename(
            {"lat_v": "lat", "lon_v": "lon"}
        )
        ds_rot =  grid_in[['cos_rot','sin_rot']]

        return ds_in_t, ds_in_u, ds_in_v, ds_rot
    
    def create_grid_out(
        self,
        lats: np.array,
        lons: np.array,
    ) -> xr.Dataset:
        """Take lat/lon of our grid and create a grid that is ready for regridder.

        Args:
            lats (np.array):
                Lats of grid_out.
            lons (xr.Dataset):
                Lons of grid_out.

        Returns:
            grid_out (xr.Dataset):
                Grid out that is ready for regridding.
        """
        grid_out = xr.Dataset()
        grid_out["lon"] = xr.DataArray(lons, dims=["lon"])
        grid_out["lat"] = xr.DataArray(lats, dims=["lat"])
        grid_out = grid_out.cf.add_bounds(["lat", "lon"])
        lat_corners = cfxr.bounds_to_vertices(
            bounds=grid_out["lat_bounds"], bounds_dim="bounds", order=None
        )
        lon_corners = cfxr.bounds_to_vertices(
            bounds=grid_out["lon_bounds"], bounds_dim="bounds", order=None
        )
        grid_out = grid_out.assign({"lat_b": lat_corners, "lon_b": lon_corners})
        grid_out = grid_out.drop_vars(["lat_bounds", "lon_bounds"])

        return grid_out
    
    def _create_regridder(self, ds_in: xr.Dataset) -> None:

        # create rotation dataset
        self.rotation_file = self.config.get("rotation_file", None)
        self.ds_rot = None
        if self.rotation_file is not None:
            mom6_grid = xr.open_dataset(self.rotation_file)
        else:
            warnings.warn(
                "MOM6Regridder._create_regridder: Could not find 'rotation_file' in configuration yaml. "
                "If using conservative regridding, it will fail. Otherwise, just vector fields will be ignored."
            )

        # create renamed datasets
        ds_in_t, ds_in_u, ds_in_v, self.ds_rot = self.create_grid_in(mom6_grid=mom6_grid)

        # create output dataset
        grid_out = self.create_grid_out(
            lats = self.lats1d_out,
            lons = self.lons1d_out,
        )

        # get nlon/nlat for input/output datsets
        nlon_i = ds_in.sizes["yh"]
        nlat_i = ds_in.sizes["xh"]
        nlon_o = len(self.lons1d_out)
        nlat_o = len(self.lats1d_out)
        self.ires = f"{nlon_i}x{nlat_i}"
        self.ores = f"{nlon_o}x{nlat_o}"

        # paths to interpolation weights files
        interp_method = self.config["interp_method"]
        wfiles = dict()
        for key in ["weights_file_t2t", "weights_file_u2t", "weights_file_v2t"]:
            vin = key[-3]
            default = f"weights-mom6-{self.ires}.C{vin}.{self.ores}.Ct.{interp_method}.nc"
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
            unmapped_to_nan=True,
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
                unmapped_to_nan=True,
            )
            reuse = os.path.exists(wfiles["weights_file_v2t"])
            self.rg_vt = xe.Regridder(
                ds_in_v,
                ds_in_t,
                interp_method,
                periodic=periodic,
                reuse_weights=reuse,
                filename=wfiles["weights_file_v2t"],
                unmapped_to_nan=True,
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
