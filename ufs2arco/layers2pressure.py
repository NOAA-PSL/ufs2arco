import os
import yaml
from typing import Optional
import numpy as np
import xarray as xr

class Layers2Pressure():
    """A class to interpolate from Lagrangian layers to pressure levels, and also compute layer thickness, etc

    Args:
        ak, bk (np.ndarray, optional): coefficients that define the vertical grid. If not provided, will
            default to the 127 vertical levels defined for NOAA GFS
        pfull, phalf (np.ndarray, optional): if ak and bk are provided, these are not necessary since they
            can be computed from those coefficients. However, these can be provided in case there are slight
            numerical differences between how these values are computed here, versus the grid information
            from an existing dataset.
        level_name, interface_name (str, optional): names to use for the vertical coordinate at
            cell center and interfaces, defaulting to "pfull" and "phalf" as in FV3. However,
            new names can be provided in order to work with datasets that use a different name.
    """
    g = 9.80665     # m / s^2
    Rd = 287.05     # J / kg / K
    Rv = 461.5      # J / kg / K
    q_min = 1e-10   # kg / kg
    z_vir = Rv / Rd - 1.
    xds = {"ak": None, "bk": None, "pfull": None, "phalf": None}

    def __init__(
        self,
        ak: Optional[np.ndarray]=None,
        bk: Optional[np.ndarray]=None,
        pfull: Optional[np.ndarray]=None,
        phalf: Optional[np.ndarray]=None,
        level_name: Optional[str]="pfull",
        interface_name: Optional[str]="phalf",
    ):

        self.level_name = level_name
        self.interface_name = interface_name

        if ak is None and bk is None:
            self.xds = self._get_default_vertical_grid()
        elif (ak is not None and bk is None) or (ak is None and bk is not None):
            raise TypeError("Must provide ak and bk, or neither, not just one.")
        else:

            newxds = self._get_xds(ak, bk, pfull, phalf)
            self.xds = newxds


    def _get_default_vertical_grid(self):

        # read vertical coordinate information
        vcoord_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_vertical_levels.yaml")
        with open(vcoord_path, "r") as f:
            vcoords = yaml.safe_load(f)

        ak = np.array(vcoords["ak"])
        bk = np.array(vcoords["bk"])
        pfull = np.array(vcoords["pfull"])
        return self._get_xds(ak, bk, pfull)


    def _get_xds(
        self,
        ak: np.ndarray,
        bk: np.ndarray,
        pfull: Optional[np.ndarray]=None,
        phalf: Optional[np.ndarray]=None,
    ) -> xr.Dataset:

        ak = ak.values if isinstance(ak, xr.DataArray) else ak
        bk = bk.values if isinstance(bk, xr.DataArray) else bk

        # Compute phalf and pfull if necessary
        if phalf is None:
            phalf = (ak + 100_000 * bk) / 100 # hPa

        if pfull is None:
            pfull = 0.5*phalf[1:] + 0.5*phalf[:-1]


        # Now convert
        phalf = xr.DataArray(
            phalf,
            coords={self.interface_name: phalf},
            dims=(self.interface_name,),
        )
        pfull = xr.DataArray(
            pfull,
            coords={self.level_name: pfull},
            dims=(self.level_name,),
        )

        ak = xr.DataArray(
            ak,
            coords={self.interface_name: phalf},
            dims=(self.interface_name,),
        )
        bk = xr.DataArray(
            bk,
            coords=ak.coords,
            dims=ak.dims,
        )

        xds = xr.Dataset({
            "ak": ak,
            "bk": bk,
            self.level_name: pfull,
            self.interface_name: phalf,
        })
        xds = xds.set_coords(["ak", "bk", self.interface_name, self.level_name])
        return xds

    @property
    def ak(self):
        return self.xds["ak"]

    @property
    def bk(self):
        return self.xds["bk"]

    @property
    def pfull(self):
        return self.xds[self.level_name]

    @property
    def phalf(self):
        return self.xds[self.interface_name]


    def calc_pressure_interfaces(self, pressfc: xr.DataArray) -> xr.DataArray:
        """Compute pressure at vertical grid interfaces

        Args:
            pressfc (xr.DataArray): surface pressure (Pa)

        Returns:
            prsi (xr.DataArray): 3D pressure field (or 4D if input has time) at vertical grid interfaces (Pa)
        """

        return self.ak.astype(pressfc.dtype) + pressfc*self.bk.astype(pressfc.dtype)

    def calc_pressure_thickness(self, prsi: xr.DataArray) -> xr.DataArray:
        """Compute pressure thickness at each vertical level

        Args:
            prsi (xr.DataArray): Pressure at vertical grid interfaces (Pa)

        Returns:
            dpres (xr.DataArray): Pressure thickness at each vertical level (Pa)
        """
        dpres = prsi.diff(self.interface_name, label="lower")
        return self._dphalf_to_pfull(dpres, name="dpres")

    def calc_dlogp(self, prsi: xr.DataArray) -> xr.DataArray:
        """Compute difference of log of interface pressure

        Args:
            prsi (xr.DataArray): Pressure at vertical grid interfaces

        Returns:
            dlogp (xr.DataArray): np.log(prsi).diff("phalf")
        """
        dlogp = np.log(prsi).diff(self.interface_name, label="lower")
        return self._dphalf_to_pfull(dlogp, name="dlogp")

    def calc_delz(
        self,
        pressfc: xr.DataArray,
        temp: xr.DataArray,
        spfh: xr.DataArray,
    ) -> xr.DataArray:
        """Compute a hydrostatic approximation of the layer thickness at each vertical level.

        Args:
            pressfc, temp, spfh (xr.DataArray): surface pressure, temperature, and specific humidity

        Returns:
            delz (xr.DataArray): layer thickness (m)
        """
        prsi = self.calc_pressure_interfaces(pressfc)
        dlogp = self.calc_dlogp(prsi)
        dlogp = dlogp.sel({self.level_name: temp[self.level_name]})
        spfh_thresh = spfh.where(spfh > self.q_min, self.q_min)
        return -self.Rd /  self.g * temp * (1. + self.z_vir*spfh_thresh) * dlogp

    def calc_layer_mean_pressure(
        self,
        pressfc: xr.DataArray,
        temp: xr.DataArray,
        spfh: xr.DataArray,
        delz: xr.DataArray,
    ) -> xr.DataArray:
        """Compute pressure at vertical grid cell center (i.e., layer mean)

        Note:
            If ``delz`` is computed by :meth:`calc_delz`, then a hydrostatic approximation is used. This may be inconsistent with the original simulation/dataset.

        Args:
            pressfc, temp, spfh, delz (xr.DataArray): surface pressure, temperature, specific humidity, and layer thickness

        Returns:
            prsl (xr.DataArray): layer mean thickness
        """

        prsi = self.calc_pressure_interfaces(pressfc)
        dpres = self.calc_pressure_thickness(prsi)
        dpres = dpres.sel({self.level_name: temp[self.level_name]})

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


    def calc_geopotential(self, hgtsfc: xr.DataArray, delz: xr.DataArray) -> xr.DataArray:
        """Compute geopotential field

        Note:
            If ``delz`` is computed by :meth:`calc_delz`, then a hydrostatic approximation is used. This may be inconsistent with the original simulation/dataset.

        Args:
            hgtsfc, delz (xr.DataArray): surface/orographic height, and vertical layer thickness

        Returns:
            geopotential (xr.DataArray): height in [m^2 / s^2] (i.e. height * gravity)
        """

        # a coordinate helper
        kp1_left = xr.DataArray(
            np.arange(len(self.pfull)),
            coords={self.level_name: self.pfull.values},
        )

        # Geopotential at the surface
        phi0 = self.g * hgtsfc
        phi0 = phi0.expand_dims({"kp1": [len(self.pfull)]})
        phi0 = phi0.reset_coords(drop=True)

        # Concatenate, cumulative sum from the ground to TOA
        dz = self.g*np.abs(delz)
        dz["kp1"] = kp1_left.sel({self.level_name: delz[self.level_name]})
        dz = dz.swap_dims({self.level_name: "kp1"}).drop_vars(self.level_name)
        dz = dz.reset_coords(drop=True)

        phii = xr.concat([dz,phi0], dim="kp1")

        phii = phii.sortby("kp1", ascending=False)
        phii = phii.cumsum("kp1")
        phii = phii.sortby("kp1", ascending=True)

        # At last, geopotential is interfacial value + .5 * layer thickness * gravity
        geopotential = phii - 0.5 * dz
        geopotential = geopotential.swap_dims({"kp1": self.level_name}).drop_vars("kp1")
        geopotential.attrs["units"] = "m**2 / s**2"
        geopotential.attrs["description"] = "Diagnosed using ufs2arco.Layers2Pressure.calc_geopotential"
        return geopotential

    def interp2pressure(
        self,
        xda: xr.DataArray,
        pstar: float,
        prsl: xr.DataArray,
        cds: Optional[xr.Dataset]=None,
    ):
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
            cds = self.get_interp_coefficients(pstar, prsl)

        xda_left = xda.where(cds["is_left"]).sum(self.level_name)
        xda_right = xda.where(cds["is_right"]).sum(self.level_name)

        result = xda_left + (xda_right - xda_left) * cds["factor"]

        mask = (cds["is_right"].sum(self.level_name) > 0) & (cds["is_left"].sum(self.level_name) > 0)
        return result.where(mask)


    def get_interp_coefficients(self, pstar: float, prsl: xr.DataArray) -> xr.Dataset:
        """Compute the coefficients needed to interpolate between pressure levels

        Args:
            pstar (float): pressure level to interpolate to in same units as prsl
            prsl (xr.DataArray): layer mean pressure

        Returns:
            cds (xr.Dataset): with coefficients needed to do interpolation
        """
        dlev = pstar - prsl
        dloglev = np.log(pstar) - np.log(prsl)

        # get coefficients representing pressure distance left and right of pstar
        is_left = dlev == dlev.where(dlev>=0).min(self.level_name)
        is_right = dlev == dlev.where(dlev<=0).max(self.level_name)

        # TODO: add extrapolation from level to ground here
        p_left = dloglev.where(is_left).sum(self.level_name)
        p_right = -dloglev.where(is_right).sum(self.level_name)

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

    def _dphalf_to_pfull(self, xda, name):
        xds = xda.to_dataset(name=name)
        xds = xds.assign_coords(self.pfull.coords)
        xds[self.level_name] = xds[self.level_name].swap_dims({self.level_name: self.interface_name})
        xds = xds.swap_dims({self.interface_name: self.level_name})
        xds = xds.drop_vars(self.interface_name)
        return xds[name]
