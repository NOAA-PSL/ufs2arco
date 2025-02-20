import logging
import fsspec

import numpy as np
import pandas as pd
import xarray as xr
import dask.array

logger = logging.getLogger("ufs2arco")

class GEFSDataset():
    """Pull GEFS data from s3://noaa-gefs-pds
    """

    static_vars = ("lsm", "orog")

    def __init__(
        self,
        t0: dict,
        fhr: dict,
        member: dict,
        chunks: dict,
        store_path: str,
    ):

        self.t0 = pd.date_range(**t0)
        assert len(fhr) == 2, \
            "GEFSDataset.__init__: 'fhr' section can only have 'start' and 'end'"
        self.fhr = np.arange(fhr["start"], fhr["end"]+1, 6)
        assert len(member) == 2, \
            "GEFSDataset.__init__: 'member' section can only have 'start' and 'end'"
        self.member = np.arange(member["start"], member["end"]+1)
        self.store_path = store_path
        self.chunks = chunks

        logger.info(str(self))

    def __len__(self) -> int:
        return len(self.t0)

    def __str__(self) -> str:
        msg = f"\n{self.name}\n"+\
            "".join(["-" for _ in range(len(self.name))]) + "\n"
        for key in ["t0", "fhr", "member", "store_path"]:
            msg += f"{key:<18s}: {getattr(self, key)}\n"
        chunkstr = "\n    ".join([f"{key:<14s}: {val}" for key, val in self.chunks.items()])
        msg += f"chunks\n    {chunkstr}"
        return msg

    @property
    def name(self):
        return f"GEFSDataset"

    def create_container(self, cache_dir="container-cache", **kwargs):
        """kwargs get passed to xr.Dataset.to_zarr
        """

        # open a minimal dataset
        xds = self.open_single_dataset(
            t0=self.t0[0],
            fhr=self.fhr[0],
            member=self.member[0],
            cache_dir=cache_dir,
        )

        # now create the t0, fhr, member dimensions
        # note that we need to handle static variables carefully
        unread_dims = {"t0": self.t0, "fhr": self.fhr, "member": self.member}
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
            [pd.Timedelta(hours=x) for x in self.fhr],
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
        logger.info(f"{self.name}.create_container: stored container at {self.store_path}\n{nds}\n")

    def open_dataset(self, cache_dir="gefs-cache"):
        """This is the naive version of the code, incorrectly assuming

        1. We can keep the whole dataset in memory
        2. We can just loop through every single time slice
        """

        dlist = []
        for date in self.t0:
            fds = self.open_single_initial_condition(date, cache_dir)
            dlist.append(fds)
        xds = xr.merge(dlist)
        return xds

    def open_single_initial_condition(self, date, cache_dir):
        """It is safe to assume we can have this in memory"""

        flist = []
        for fhr in self.fhr:
            logger.info(f"Reading {date}, {fhr}h for members {self.member[0]} - {self.member[-1]}")
            single_time_slice = xr.merge(
                [
                    self.open_single_dataset(
                        date=date,
                        fhr=fhr,
                        member=member,
                        cache_dir=cache_dir,
                    )
                    for member in self.member
                ],
            )
            flist.append(single_time_slice)
        return xr.merge(flist)


    def open_single_dataset(self, t0, fhr, member, cache_dir):
        """Assume for now:
        1. we are reading from both the a and b files
        2. we are caching both of the files locally
        """

        # 1. cache the grib files for this date, member, fhr
        cached_files = {}
        for k in ["a", "b"]:
            path = self.build_path(
                t0=t0,
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
                    f"{self.name}: File Not Found: {path}\n\t" +\
                    f"(t0, member, fhr, key) = {t0} {member} {fhr} {k}"
                )
            except:
                local_file = None
                logger.warning(
                    f"{self.name}: Trouble finding the file: {path}\n\t" +\
                    f"(t0, member, fhr, key) = {t0} {member} {fhr} {k}"
                )

            cached_files[k] = local_file

        # 2. read data arrays from those files
        dsdict = {}
        if cached_files["a"] is not None and cached_files["b"] is not None:

            is_static = fhr == 0 and member == 0 and t0 == self.t0[0]
            read_dict = self._ic_variables if is_static else self._fc_variables
            for varname, open_kwargs in read_dict.items():
                dslist = []
                for a_or_b in open_kwargs["param_list"]:
                    try:
                        thisvar = self.open_single_variable(
                            file=cached_files[a_or_b],
                            varname=varname,
                            member=member,
                            filter_by_keys=open_kwargs["filter_by_keys"],
                        )
                    except:
                        thisvar = None
                        logger.warning(
                            f"{self.name}: Trouble opening {varname}\n\t" +\
                            f"(t0, member, fhr, key) = {t0} {member} {fhr} {a_or_b}"
                        )
                    dslist.append(thisvar)
                if not any(x is None for x in dslist):
                    dsdict[varname] = xr.merge(dslist)[varname]
                else:
                    dsdict[varname] = xr.Dataset()

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
            for key in ["step", "time", "valid_time"]:
                if key in xds:
                    xds = xds.drop_vars(key)

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
            valid_time = xds["t0"] + xds["lead_time"]
            if "valid_time" in xds:

                xds["valid_time"] = xds["valid_time"].expand_dims(["t0", "fhr"])
                assert valid_time.squeeze() == xds.valid_time.squeeze()
                xds = xds.drop_vars("valid_time")

            xds["valid_time"] = valid_time
            xds = xds.set_coords("valid_time")

        return xds[varname]

    def build_path(self, t0, member, fhr, a_or_b):
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
