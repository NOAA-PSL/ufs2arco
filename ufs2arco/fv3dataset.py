import os
import yaml
from typing import Dict, List, Callable
from datetime import datetime
import numpy as np
import xarray as xr

from .ufsdataset import UFSDataset

class FV3Dataset(UFSDataset):
    __doc__ = UFSDataset.__doc__
    def __init__(self, path_in: Callable, config_filename: str, is_nested: bool = False) -> None:
        super(FV3Dataset, self).__init__(path_in, config_filename, is_nested)
        self.zarr_name = "fv3.zarr"
        self.chunks_in = self.chunks_in if len(self.chunks_in) != 0 else {
            "pfull": 1,
            "grid_yt": -1,
            "grid_xt": -1,
        }

        self.chunks_out = self.chunks_out if len(self.chunks_out) != 0 else {
            "time": 1,
            "pfull": 1,
            "grid_yt": -1,
            "grid_xt": -1,
        }

    def open_dataset(self, cycles: datetime, fsspec_kwargs=None, **kwargs):
        xds = super().open_dataset(cycles, fsspec_kwargs, **kwargs)

        # Deal with time
        xds = xds.rename({"time": "cftime"})
        xds["time"] = self._cftime2time(xds["cftime"])
        xds["ftime"] = self._time2ftime(xds["time"], cycles)
        xds = xds.swap_dims({"cftime": "time"})
        xds = xds.set_coords(["time", "cftime", "ftime"])

        # convert ak/bk attrs to coordinate arrays
        for key in ["ak", "bk"]:
            if key in xds.attrs:
                xds[key] = xr.DataArray(
                    xds.attrs.pop(key),
                    coords=xds["phalf"].coords,
                    dims=xds["phalf"].dims,
                )
                xds = xds.set_coords(key)

        # rename grid_yt.long_name to avoid typo
        xds["grid_yt"].attrs["long_name"] = "T-cell latitude"
        return xds


class Layers2Pressure():
    """A class to interpolate from Lagrangian layers to pressure levels, and also compute layer thickness, etc"""
    g = 9.80665     # m / s^2
    Rd = 287.05     # J / kg / K
    Rv = 461.5      # J / kg / K
    q_min = 1e-10   # kg / kg
    z_vir = Rv / Rd - 1.

    def __init__(self):

        # read vertical coordinate information
        vcoord_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_vertical_levels.yaml")
        with open(vcoord_path, "r") as f:
            vcoords = yaml.safe_load(f)

        k = np.arange(len(vcoords["pfull"]))
        kp1 = np.arange(len(vcoords["ak"]))
        self.ak = xr.DataArray(
            vcoords["ak"],
            coords={"kp1": kp1},
            dims=("kp1",),
            name="ak",
        )
        self.bk = xr.DataArray(
            vcoords["bk"],
            coords=self.ak.coords,
            dims=self.ak.dims,
            name="bk",
        )
        self.level = xr.DataArray(
            vcoords["pfull"],
            coords={"k": k},
            dims=("k",),
            name="level",
        )

    def calc_pressure_interfaces(self, pressfc) -> xr.DataArray:
        """Compute pressure at vertical grid interfaces

        Args:
            pressfc (xr.DataArray): surface pressure (Pa)

        Returns:
            prsi (xr.DataArray): 3D pressure field (or 4D if input has time) at vertical grid interfaces (Pa)
        """

        return self.ak.astype(pressfc.dtype) + pressfc*self.bk.astype(pressfc.dtype)

    def calc_pressure_thickness(self, prsi) -> xr.DataArray:
        """Compute pressure thickness at each vertical level

        Args:
            prsi (xr.DataArray): Pressure at vertical grid interfaces (Pa)

        Returns:
            dpres (xr.DataArray): Pressure thickness at each vertical level (Pa)
        """
        dpres = prsi.diff("kp1", label="lower")
        return self._dkp1_to_level(dpres)

    def calc_dlogp(self, prsi) -> xr.DataArray:
        """Compute difference of log of interface pressure

        Args:
            prsi (xr.DataArray): Pressure at vertical grid interfaces

        Returns:
            dlogp (xr.DataArray): np.log(prsi).diff("kp1")
        """
        dlogp = np.log(prsi).diff("kp1", label="lower")
        return self._dkp1_to_level(dlogp)

    def calc_delz(self, pressfc, temp, spfh) -> xr.DataArray:
        """Compute layer thickness at each vertical level

        Args:
            pressfc, temp, spfh (xr.DataArray): surface pressure, temperature, and specific humidity

        Returns:
            delz (xr.DataArray): layer thickness (m)
        """
        prsi = self.calc_pressure_interfaces(pressfc)
        dlogp = self.calc_dlogp(prsi)
        dlogp = dlogp.sel(level=temp["level"])
        spfh_thresh = spfh.where(spfh > self.q_min, self.q_min)
        return -self.Rd /  self.g * temp * (1. + self.z_vir*spfh_thresh) * dlogp

    def calc_layer_mean_pressure(self, pressfc, temp, spfh, delz) -> xr.DataArray:
        """Compute pressure at vertical grid cell center (i.e., layer mean)

        Args:
            pressfc, temp, spfh, delz (xr.DataArray): surface pressure, temperature, specific humidity, and layer thickness

        Returns:
            prsl (xr.DataArray): layer mean thickness
        """

        prsi = self.calc_pressure_interfaces(pressfc)
        dpres = self.calc_pressure_thickness(prsi)
        dpres = dpres.sel(level=temp["level"])

        # rTv computed here:
        # https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere/blob/ab195d5026ca4c221b6cbb3888c8ae92d711f89a/driver/fvGFS/atmosphere.F90#L2199-L2200
        spfh_thresh = spfh.where(spfh > self.q_min, self.q_min)
        rTv = self.Rd * temp * (1. + self.z_vir*spfh_thresh)

        # layer mean pressure computed here:
        # https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere/blob/ab195d5026ca4c221b6cbb3888c8ae92d711f89a/driver/fvGFS/atmosphere.F90#L2206-L2209
        # Numerator, our dpres = prsl in fortran code, as shown here: https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere/blob/ab195d5026ca4c221b6cbb3888c8ae92d711f89a/driver/fvGFS/atmosphere.F90#L2122
        # Denominator is phii(k+1) - phii(k), which is equal to -g*delz,
        # see https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere/blob/ab195d5026ca4c221b6cbb3888c8ae92d711f89a/driver/fvGFS/atmosphere.F90#L2125-L2131
        return dpres*rTv / (-self.g*delz)


    def calc_geopotential(self, hgtsfc, delz):
        """Note delz has to have vertical coordinate named level not pfull"""

        # a coordinate helper
        kp1_left = xr.DataArray(
            np.arange(len(self.level)),
            coords={"level": self.level.values},
        )

        # Geopotential at the surface
        phi0 = self.g * hgtsfc
        phi0 = phi0.expand_dims({"kp1": [len(self.level)]})

        # Concatenate, cumulative sum from the ground to TOA
        dz = self.g*np.abs(delz)
        dz["kp1"] = kp1_left.sel(level=delz["level"])
        dz = dz.swap_dims({"level": "kp1"}).drop_vars("level")

        phii = xr.concat([dz,phi0], dim="kp1")

        phii = phii.sortby("kp1", ascending=False)
        phii = phii.cumsum("kp1")
        phii = phii.sortby("kp1", ascending=True)

        # At last, geopotential is interfacial value + .5 * layer thickness * gravity
        geopotential = phii - 0.5 * dz
        geopotential = geopotential.swap_dims({"kp1": "level"}).drop_vars("kp1")
        geopotential.attrs["units"] = "m^2 / s^2"
        return geopotential

    def interp2pressure(self, xda, pstar, prsl, cds=None):
        """Interpolate data on FV3 native vertical grid to pressure level (p*)

        Args:
            xda (xr.DataArray): field to be interpolated
            pstar (float): pressure level to be interpolated to, same units as prsl please
            prsl (xr.DataArray): layer mean pressure
            cds (xr.Dataset, optional): dataset with coefficients needed to do interpolation

        Returns:
            pda (xr.DataArray): field interpolated to pressure pstar
        """

        if cds is None:
            cds = _get_interp_coefficients(pstar, prsl)

        xda_left = xda.where(cds["is_left"]).sum("level")
        xda_right = xda.where(cds["is_right"]).sum("level")

        return xda_left + (xda_right - xda_left) * cds["factor"]


    @staticmethod
    def _get_interp_coefficients(pstar: float, prsl: xr.DataArray) -> xr.Dataset:
        dlev = pstar - prsl
        dloglev = np.log(pstar) - np.log(prsl)

        # get coefficients representing pressure distance left and right of pstar
        is_left = dlev == dlev.where(dlev>=0).min("level")
        is_right = dlev == dlev.where(dlev<=0).max("level")

        # TODO: add extrapolation from level to ground here
        p_left = dloglev.where(is_left).sum("level")
        p_right = -dloglev.where(is_right).sum("level")

        denominator = p_left + p_right
        denominator = denominator.where(denominator > 1e-6, 1e-6)
        factor = p_left / denominator
        factor = factor.where(
            factor < 10.,
            10.
        ).where(
            factor > -10.,
            -10.,
        )
        return xr.Dataset({
            "is_left": is_left,
            "is_right": is_right,
            "p_left": p_left,
            "p_right": p_right,
            "denominator": denominator,
            "factor": factor,
        })




    def _dkp1_to_level(self, xda):
        xda = xda.rename({"kp1": "k"})
        xda["level"] = self.level
        xda = xda.swap_dims({"k": "level"})
        xda = xda.drop_vars({"k"})
        return xda
