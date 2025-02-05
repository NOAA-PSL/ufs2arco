import yaml
import logging
import fsspec

import numpy as np
import pandas as pd
import xarray as xr
import dask.array

logger = logging.getLogger("ufs2arco")

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

    static_vars = ("lsm", "orog")

    def __init__(
        self,
        config_filename: str,
    ):

        with open(config_filename, "r") as f:
            contents = yaml.safe_load(f)
            self.config = contents


        self.dates = pd.date_range(**self.config["dates"])
        self.fhrs = self.config.get("fhrs", (0, 6))
        self.members = np.arange(
            self.config["members"].get("start", 0),
            self.config["members"].get("end", 1)+1,
        )
        self.store_path = self.config["store_path"]
        if "chunks" not in self.config:
            self.chunks = {
                "t0": 1,
                "fhr": 1,
                "member": 1,
                "pressure": 1,
                "latitude": -1,
                "longitude": -1,
            }
        else:
            self.chunks = self.config["chunks"]

        logger.info(str(self))

    def __len__(self) -> int:
        return len(self.dates)

    def __str__(self) -> str:
        msg = f"\n{self._name}\n"+\
            "".join(["-" for _ in range(len(self._name))]) + "\n"
        for key in ["dates", "fhrs", "members", "store_path"]:
            msg += f"{key:<18s}: {getattr(self, key)}\n"
        chunkstr = "\n    ".join([f"{key:<14s}: {val}" for key, val in self.chunks.items()])
        msg += f"chunks\n    {chunkstr}"
        return msg

    @property
    def _name(self):
        return f"GEFSDataset"

    def create_container(self, cache_dir="container-cache", **kwargs):
        """kwargs get passed to xr.Dataset.to_zarr
        """

        # open a minimal dataset
        xds = self.open_single_dataset(
            date=self.dates[0],
            fhr=self.fhrs[0],
            member=self.members[0],
            cache_dir=cache_dir,
        )

        # now create the t0, fhr, member dimensions
        # note that we need to handle static variables carefully
        unread_dims = {"t0": self.dates, "fhr": self.fhrs, "member": self.members}
        extra_coords = ["lead_time", "valid_time"]

        # create container
        nds = xr.Dataset()
        for key, array in unread_dims.items():
            nds[key] = xr.DataArray(
                array,
                coords={key: array},
                dims=key,
                attrs=xds[key].attrs.copy(),
            )
        for key in xds.dims:
            if key not in unread_dims.keys():
                nds[key] = xds[key].copy()

        # compute the full version of extra_coords
        nds["lead_time"] = xr.DataArray(
            [pd.Timedelta(hours=x) for x in self.fhrs],
            coords=nds["fhr"].coords,
            attrs=xds["lead_time"].attrs.copy(),
        )

        nds["valid_time"] = nds["t0"] + nds["lead_time"]
        nds["valid_time"].attrs = xds["valid_time"].attrs.copy()
        nds = nds.set_coords(extra_coords)

        # squeeze what we've read
        xds = xds.squeeze().drop_vars(list(unread_dims.keys())+extra_coords)

        # create container arrays
        for varname in xds.data_vars:
            if varname not in self.static_vars:
                dims = tuple(unread_dims.keys()) + xds[varname].dims
                shape = tuple(len(nds[d]) for d in unread_dims.keys()) + xds[varname].shape
                chunks = {list(dims).index(key): self.chunks[key] for key in dims}
                nds[varname] = xr.DataArray(
                    data=dask.array.zeros(
                        shape=shape,
                        chunks=chunks,
                        dtype=xds[varname].dtype,
                    ),
                    dims=dims,
                    attrs=xds[varname].attrs.copy(),
                )
            else:
                nds[varname] = xds[varname].copy()

        nds.to_zarr(self.store_path, compute=False, **kwargs)
        logger.info(f"{self._name}.create_container: stored container at {self.store_path}")


    def find_my_region(self, xds):
        """Given a dataset, that's assumed to be a subset of the initial dataset,
        find the logical index values where this should be stored in the final zarr store

        Args:
            xds (xr.Dataset): with a subset of the data (i.e., a couple of initial conditions)

        Returns:
            region (dict): indicating the zarr region to store in, based on the initial condition indices
        """
        batch_dates = [pd.Timestamp(t0) for t0 in xds["t0"].values]
        date_indices = [list(self.dates).index(date) for date in batch_dates]

        region = {k: slice(None, None) for k in xds.dims}
        region["t0"] = slice(date_indices[0], date_indices[-1]+1)
        return region


    def open_dataset(self, cache_dir="gefs-cache"):
        """This is the naive version of the code, incorrectly assuming

        1. We can keep the whole dataset in memory
        2. We can just loop through every single time slice
        """

        dlist = []
        for date in self.dates:
            fds = self.open_single_initial_condition(date, cache_dir)
            dlist.append(fds)
        xds = xr.merge(dlist)
        return xds

    def open_single_initial_condition(self, date, cache_dir):
        """It is safe to assume we can have this in memory"""

        flist = []
        for fhr in self.fhrs:
            logger.info(f"Reading {date}, {fhr}h for members {self.members[0]} - {self.members[-1]}")
            single_time_slice = xr.merge(
                [
                    self.open_single_dataset(
                        date=date,
                        fhr=fhr,
                        member=member,
                        cache_dir=cache_dir,
                    )
                    for member in self.members
                ],
            )
            flist.append(single_time_slice)
        return xr.merge(flist)


    def open_single_dataset(self, date, fhr, member, cache_dir):
        """Assume for now:
        1. we are reading from both the a and b files
        2. we are caching both of the files locally
        """

        # 1. cache the grib files for this date, member, fhr
        cached_files = {}
        for k in ["a", "b"]:
            path = self.build_path(
                date=date,
                member=member,
                fhr=fhr,
                a_or_b=k,
            )
            try:
                local_file = fsspec.open_local(
                    path,
                    s3={"anon": True},
                    filecache={"cache_storage": cache_dir},
                )
            except FileNotFoundError:
                local_file = None
                logger.warning(
                    f"{self._name}: File Not Found: {path}\n\t" +\
                    f"(date, member, fhr, key) = {date} {member} {fhr} {k}"
                )

            cached_files[k] = local_file

        # 2. read data arrays from those files
        dsdict = {}
        if cached_files["a"] is not None and cached_files["b"] is not None:

            is_static = fhr == 0 and member == 0 and date == self.dates[0]
            read_dict = self._ic_variables if is_static else self._fc_variables
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
            xds = xds.drop_vars(["step", "time", "valid_time"])

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
            xds = xds.swap_dims({"lead_time": "fhr"})
            xds["valid_time"] = xds["valid_time"].expand_dims(["t0", "fhr"])

        return xds[varname]

    def build_path(self, date, member, fhr, a_or_b):
        c_or_p = "c" if member == 0 else "p"
        bucket = f"s3://noaa-gefs-pds"
        outer = f"gefs.{date.year:04d}{date.month:02d}{date.day:02d}/{date.hour:02d}"

        # Thanks to Herbie for figuring these out
        if date < pd.Timestamp("2017-07-27"):
            middle = ""
            fname = f"ge{c_or_p}{member:02d}.t{date.hour:02d}z.pgrb2{a_or_b}f{fhr:03d}"

        elif date < pd.Timestamp("2020-09-24"):
            middle = f"pgrb2{a_or_b}/"
            fname = f"ge{c_or_p}{member:02d}.t{date.hour:02d}z.pgrb2{a_or_b}f{fhr:02d}"

        else:
            middle = f"atmos/pgrb2{a_or_b}p5/"
            fname = f"ge{c_or_p}{member:02d}.t{date.hour:02d}z.pgrb2{a_or_b}.0p50.f{fhr:03d}"

        fullpath = f"filecache::{bucket}/{outer}/{middle}{fname}"
        logger.debug(f"GEFSDataset.build_path: reading {fullpath}")
        return fullpath



    @property
    def _ic_variables(self):
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
        fckw = self._ic_variables.copy()
        for key in self.static_vars:
            fckw.pop(key)
        return fckw
