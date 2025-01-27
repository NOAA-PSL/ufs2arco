import yaml
import logging
import fsspec

import numpy as np
import pandas as pd
import xarray as xr

from .log import setup_simple_log


class GEFSDataset():
    """Pull GEFS data from s3://noaa-gefs-pds

    Here are the full options

    dates:
      start: 2017-01-01T00
      end: ... basically current date

    fhrs: [0, 6]
    ... need to fill this out

    TODO:
        * fhrs as start/end since there are many
        * figure out the threshold dates where bucket format changes
        * add a create_container method along with store_path attr

    """

    def __init__(
        self,
        config_filename: str,
    ):
        setup_simple_log()

        with open(config_filename, "r") as f:
            contents = yaml.safe_load(f)
            self.config = contents


        self.dates = pd.date_range(**self.config["dates"])
        self.fhrs = self.config.get("fhrs", (0, 6))
        self.members = np.arange(
            self.config["members"].get("start", 0),
            self.config["members"].get("end", 1)+1,
        )

        logging.info(f"{self.config}")
        logging.info(f"Dates:\n{self.dates}\n")
        logging.info(f"Forecast Hours:\n{self.fhrs}\n")
        logging.info(f"Members:\n{self.members}\n")

    def __len__(self) -> int:
        return len(self.dates)

    def open_dataset(self, cache_dir="./gribcache"):
        """This is the naive version of the code, incorrectly assuming

        1. We can keep the whole dataset in memory
        2. We can just loop through every single time slice
        """

        dlist = []
        for date in self.dates:
            fds = self.open_single_initial_condition(date, cache_dir=cache_dir)
            dlist.append(fds)
        xds = xr.merge(dlist)
        return xds

    def open_single_initial_condition(self, date, cache_dir="./gribcache"):
        """It is safe to assume we can have this in memory"""

        flist = []
        for fhr in self.fhrs:
            logging.info(f"Reading {date}, {fhr}h for members {self.members[0]} - {self.members[-1]}")
            single_time_slice = xr.merge(
                [
                    self.open_single_dataset(
                        date=date,
                        fhr=fhr,
                        member=member,
                    )
                    for member in self.members
                ],
            )
            flist.append(single_time_slice)
        return xr.merge(flist)


    def open_single_dataset(self, date, fhr, member, cache_dir="./gribcache"):
        """Assume for now:
        1. we are reading from both the a and b files
        2. we are caching both of the files locally
        """

        # 1. cache the grib files for this date, member, fhr
        cached_files = {
           k: fsspec.open_local(
                self.build_path(
                    date=date,
                    member=member,
                    fhr=fhr,
                    a_or_b=k,
                ),
                s3={"anon": True},
                filecache={"cache_storage": cache_dir},
            )
            for k in ["a", "b"]
        }

        # 2. read data arrays from those files
        dsdict = {}
        read_dict = self._ic_variables if fhr == 0 and member == 0 and date == self.dates[0] else self._fc_variables
        for varname, open_kwargs in read_dict.items():
            dslist = [
                self.open_single_variable(
                    file=cached_files[a_or_b],
                    varname=varname,
                    member=member,
                    filter_by_keys=open_kwargs["filter_by_keys"],
                )
                for a_or_b in open_kwargs["param_list"]
            ]
            dsdict[varname] = xr.merge(dslist)[varname]

        xds = xr.Dataset(dsdict)
        return xds


    def open_single_variable(self, file, varname, member, filter_by_keys=None):
        xds = xr.open_dataset(file, engine="cfgrib", filter_by_keys=filter_by_keys)
        xda = xds[varname]
        if "isobaricInhPa" in xds.coords:
            if len(xda.dims) < 3:
                vv = xda["isobaricInhPa"].values
                xda = xda.expand_dims({"isobaricInhPa": [vv]})
            xda = xda.rename({"isobaricInhPa": "pressure"})

        for v in ["heightAboveGround", "number", "surface"]:
            if v in xda.coords:
                xda = xda.drop_vars(v)

        xds = xda.to_dataset(name=varname)
        if varname in ["lsm", "orog"]:
            xds = xds.drop_vars(["step", "time"])

        else:
            xds = xds.rename(
                {
                    "time": "t0",
                    "step": "lead_time",
                },
            )
            xds = xds.expand_dims(["t0", "lead_time", "member"])
            xds["fhr"] = xr.DataArray(
                int(xds["lead_time"].values / 1e9 / 3600),
                coords=xds["lead_time"].coords,
                attrs={
                    "long_name": "hours since initial time",
                    "units": "hours",
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
            xds = xds.swap_dims({"lead_time": "fhr"})
            xds["valid_time"] = xds["valid_time"].expand_dims(["t0", "fhr"])

        return xds[varname]

    def build_path(self, date, member, fhr, a_or_b):
        c_or_p = "c" if member == 0 else "p"
        bucket = f"s3://noaa-gefs-pds"
        outer = f"gefs.{date.year:04d}{date.month:02d}{date.day:02d}/{date.hour:02d}"

        missing_dates = (
            pd.Timestamp("2020-09-23T06"),
        )
        if date in missing_dates:
            raise ValueError("GEFSDataset.build_path: {date} is missing from the GEFS archive")

        # Thanks to Herbie for figuring these out
        if date < pd.Timestamp("2017-07-27"):
            middle = ""
            fname = f"ge{c_or_p}{member:02d}.t{date.hour:02d}z.pgrb2{a_or_b}f{fhr:03d}"

        elif date < pd.Timestamp("2020-09-23"):
            middle = f"pgrb2{a_or_b}/"
            fname = f"ge{c_or_p}{member:02d}.t{date.hour:02d}z.pgrb2{a_or_b}f{fhr:02d}"

        else:
            middle = f"atmos/pgrb2{a_or_b}p5/"
            fname = f"ge{c_or_p}{member:02d}.t{date.hour:02d}z.pgrb2{a_or_b}.0p50.f{fhr:03d}"

        fullpath = f"filecache::{bucket}/{outer}/{middle}{fname}"
        logging.debug(f"GEFSDataset.build_path: reading {fullpath}")
        return fullpath



    @property
    def _ic_variables(self):
        return {
            "lsm": {
                "rename": "land_static",
                "param_list": ["b"],
                "filter_by_keys": {
                    "typeOfLevel": "surface",
                    "paramId": [172],
                },
            },
            "orog": {
                "rename": "land_static",
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
        fckw = self._ic_variables.copy()
        fckw.pop("lsm")
        fckw.pop("orog")
        return fckw
