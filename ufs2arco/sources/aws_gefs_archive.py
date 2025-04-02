import logging
from typing import Optional
import fsspec

import pandas as pd
import xarray as xr

from ufs2arco.sources import EnsembleForecastSource

logger = logging.getLogger("ufs2arco")

class AWSGEFSArchive(EnsembleForecastSource):
    """Access NOAA's forecast archive from the Global Ensemble Forecast System (GEFS) at s3://noaa-gefs-pds."""

    static_vars = ("lsm", "orog")
    sample_dims = ("t0", "fhr", "member")
    base_dims = ("latitude", "longitude")

    @property
    def available_variables(self) -> tuple:
        return tuple(self._ic_variables.keys())

    @property
    def available_levels(self) -> tuple:
        return (
            10, 20, 30, 50, 70,
            100, 150, 200, 250, 300, 350, 400, 450,
            500, 550, 600, 650, 700, 750, 800, 850,
            900, 925, 950, 975, 1000,
        )

    @property
    def rename(self) -> dict:
        return {
            "time": "t0",
            "step": "lead_time",
            "isobaricInhPa": "level",
        }

    def open_sample_dataset(
        self,
        t0: pd.Timestamp,
        fhr: int,
        member: int,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:

        # 1. cache the grib files for this date, member, fhr
        cached_files = {}
        for k in ["a", "b"]:
            path = self._build_path(
                t0=t0,
                member=member,
                fhr=fhr,
                a_or_b=k,
            )
            if cache_dir is not None:
                try:
                    local_file = fsspec.open_local(
                        path,
                        s3={"anon": True},
                        filecache={"cache_storage": cache_dir},
                    )
                except FileNotFoundError:
                    local_file = None
                    logger.warning(
                        f"{self.name}: File Not Found: {path}\n\t" +
                        f"(t0, member, fhr, key) = {t0} {member} {fhr} {k}"
                    )
                except:
                    local_file = None
                    logger.warning(
                        f"{self.name}: Trouble finding the file: {path}\n\t" +
                        f"(t0, member, fhr, key) = {t0} {member} {fhr} {k}"
                    )
                cached_files[k] = local_file
            else:
                cached_files[k] = path

        # 2. read data arrays from those files
        dsdict = {}
        osv = open_static_vars or self._open_static_vars(t0, fhr, member)
        read_dict = self._ic_variables if osv else self._fc_variables
        read_dict = {k: v for k,v in read_dict.items() if k in self.variables}
        if cached_files["a"] is not None and cached_files["b"] is not None:
            for varname, open_kwargs in read_dict.items():
                dslist = []
                for a_or_b in open_kwargs["param_list"]:
                    try:
                        thisvar = self._open_single_variable(
                            file=cached_files[a_or_b],
                            varname=varname,
                            member=member,
                            filter_by_keys=open_kwargs["filter_by_keys"],
                        )
                    except:
                        thisvar = None
                        logger.warning(
                            f"{self.name}: Trouble opening {varname}\n\t" +
                            f"(t0, member, fhr, key) = {t0} {member} {fhr} {a_or_b}"
                        )
                    dslist.append(thisvar)
                if not any(x is None for x in dslist):
                    dsdict[varname] = xr.merge(dslist)[varname]
                else:
                    dsdict[varname] = xr.DataArray(name=varname)
        xds = xr.Dataset(dsdict)
        xds = self.apply_slices(xds)
        return xds

    def _open_single_variable(
        self,
        file: fsspec.spec.AbstractFileSystem,
        varname: str,
        member: int,
        filter_by_keys: dict,
    ) -> xr.DataArray:
        """
        Open a single variable from a GRIB file.

        Args:
            file (fsspec.spec.AbstractFileSystem): The file to read.
            varname (str): The variable name to extract.
            member (int): The ensemble member ID.
            filter_by_keys (dict): Keys to filter the variable by.

        Returns:
            xr.DataArray: The extracted variable as an xarray DataArray.
        """
        xds = xr.open_dataset(file, engine="cfgrib", filter_by_keys=filter_by_keys)
        xda = xds[varname]

        if "isobaricInhPa" in xds.coords:
            if len(xda.dims) < 3:
                vv = xda["isobaricInhPa"].values
                xda = xda.expand_dims({"isobaricInhPa": [vv]})

        for v in ["heightAboveGround", "number", "surface"]:
            if v in xda.coords:
                xda = xda.drop_vars(v)


        xds = xda.to_dataset(name=varname)
        for key, val in self.rename.items():
            if key in xds:
                xds = xds.rename({key: val})
        if varname in self.static_vars:
            for key in ["lead_time", "t0", "valid_time"]:
                if key in xds:
                    xds = xds.drop_vars(key)
        else:

            if "level" in xds and self.levels is not None:
                level_selection = [l for l in self.levels if l in xds.level.values]
                if len(level_selection) == 0:
                    return xr.DataArray(name=varname)
                xds = xds.sel(level=level_selection, **self._level_sel_kwargs)

            xds = xds.expand_dims(["t0", "lead_time", "member"])
            xds["fhr"] = xr.DataArray(
                int(xds["lead_time"].values / 1e9 / 3600),
                coords=xds["lead_time"].coords,
                attrs={
                    "long_name": "hours since initial time",
                    "units": "integer hours",
                },
            )
            xds["member"] = xr.DataArray(
                [member],
                coords={"member": [member]},
                dims=("member",),
                attrs={
                    "description": "ID=0 comes from gecXX files, ID>0 comes from gepXX files",
                    "long_name": "ensemble member ID",
                },
            )

            # recreate valid_time, since it's not always there
            xds = xds.swap_dims({"lead_time": "fhr"})
            valid_time = xds["t0"] + xds["lead_time"]
            if "valid_time" in xds:
                xds["valid_time"] = xds["valid_time"].expand_dims(["t0", "fhr"])
                assert valid_time.squeeze() == xds.valid_time.squeeze()
                xds = xds.drop_vars("valid_time")

            xds["valid_time"] = valid_time
            xds = xds.set_coords("valid_time")
        return xds[varname]


    def _build_path(self, t0: pd.Timestamp, member: int, fhr: int, a_or_b: str) -> str:
        """
        Build the file path to a GRIB file based on the provided parameters.

        Args:
            t0 (pd.Timestamp): The initial condition timestamp.
            member (int): The ensemble member ID.
            fhr (int): The forecast hour.
            a_or_b (str): The file type, either 'a' or 'b'.

        Returns:
            str: The constructed file path.
        """
        c_or_p = "c" if member == 0 else "p"
        bucket = f"s3://noaa-gefs-pds"
        outer = f"gefs.{t0.year:04d}{t0.month:02d}{t0.day:02d}/{t0.hour:02d}"
        # Thanks to Herbie for figuring these out
        if t0 < pd.Timestamp("2018-07-27"):
            middle = ""
            fname = f"ge{c_or_p}{member:02d}.t{t0.hour:02d}z.pgrb2{a_or_b}f{fhr:03d}"
        elif t0 < pd.Timestamp("2020-09-23T12"):
            middle = f"pgrb2{a_or_b}/"
            fname = f"ge{c_or_p}{member:02d}.t{t0.hour:02d}z.pgrb2{a_or_b}f{fhr:02d}"
        else:
            middle = f"atmos/pgrb2{a_or_b}p5/"
            fname = f"ge{c_or_p}{member:02d}.t{t0.hour:02d}z.pgrb2{a_or_b}.0p50.f{fhr:03d}"
        fullpath = f"filecache::{bucket}/{outer}/{middle}{fname}"
        logger.debug(f"{self.name}._build_path: reading {fullpath}")
        return fullpath

    @property
    def _ic_variables(self) -> dict:
        """
        Get the dictionary of initial condition variables.

        Returns:
            dict: The dictionary of variables for initial conditions.
        """
        return {
            "lsm": {
                "param_list": ["b"],
                "filter_by_keys": {
                    "typeOfLevel": "surface",
                    "paramId": [172],
                },
            },
            "orog": {
                "param_list": ["a"],
                "filter_by_keys": {
                    "typeOfLevel": "surface",
                    "paramId": [228002],
                },
            },
            "sp": {
                "param_list": ["a"],
                "filter_by_keys": {
                    "typeOfLevel": "surface",
                    "paramId": [134],
                },
            },
            "u10": {
                "param_list": ["a"],
                "filter_by_keys": {
                    "typeOfLevel": "heightAboveGround",
                    "paramId": [165],
                },
            },
            "v10": {
                "param_list": ["a"],
                "filter_by_keys": {
                    "typeOfLevel": "heightAboveGround",
                    "paramId": [166],
                },
            },
            "t2m": {
                "param_list": ["a"],
                "filter_by_keys": {
                    "typeOfLevel": "heightAboveGround",
                    "paramId": [167],
                },
            },
            "sh2": {
                "param_list": ["b"],
                "filter_by_keys": {
                    "typeOfLevel": "heightAboveGround",
                    "paramId": [174096],
                },
            },
            "gh": {
                "param_list": ["a", "b"],
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "paramId": [156],
                },
            },
            "u": {
                "param_list": ["a", "b"],
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "paramId": [131],
                },
            },
            "v": {
                "param_list": ["a", "b"],
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "paramId": [132],
                },
            },
            "w": {
                "param_list": ["a", "b"],
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "paramId": [135],
                },
            },
            "t": {
                "param_list": ["a", "b"],
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "paramId": [130],
                },
            },
            "q": {
                "param_list": ["b"],
                "filter_by_keys": {
                    "typeOfLevel": "isobaricInhPa",
                    "paramId": [133],
                },
            },
        }

    @property
    def _fc_variables(self):
        """
        Get the dictionary of forecast variables, which is notably different from
        the initial condition files.

        Returns:
            dict: The dictionary of variables for forecast files/variables
        """
        fckw = self._ic_variables.copy()
        for key in self.static_vars:
            fckw.pop(key)
        return fckw
