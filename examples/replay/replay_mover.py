from os.path import join, isdir
from datetime import datetime, timedelta
import yaml
from shutil import rmtree
import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as darray
from zarr import NestedDirectoryStore

from ufs2arco import FV3Dataset

from timer import Timer

class ReplayMover1Degree():
    """

    Note:
        Currently this makes the unnecessary but easy-to-implement assumption that we want forecast_hours 0 & 3.
        This assumption is key to the hard coded end date and timedelta used to make :attr:`xtime`.
        It should also be confirmed that this assumption does not impact the "ftime" variable, that is right now
        being created in the container dataset. The values should be overwritten when the data are actually
        generated, but who knows.
    """


    n_jobs = None
    n_cycles = None

    forecast_hours = None
    file_prefixes = None

    @property
    def xcycles(self):
        cycles = pd.date_range(start="1994-01-01", end="1999-06-13T06:00:00", freq="6h")
        return xr.DataArray(cycles, coords={"cycles": cycles}, dims="cycles")


    @property
    def xtime(self):
        time = pd.date_range(start="1994-01-01", end="1999-06-13T09:00:00", freq="3h")
        iau_time = time - timedelta(hours=6)
        return xr.DataArray(iau_time, coords={"time": iau_time}, dims="time", attrs={"long_name": "time", "axis": "T"})


    @property
    def splits(self):
        """The indices used to split all cycles across :attr:`n_jobs`"""
        return [int(x) for x in np.linspace(0, len(self.xcycles), self.n_jobs+1)]

    def cache_storage(self, job_id):
        return f"{self.main_cache_path}/{job_id}"

    def ods_kwargs(self, job_id):
        okw = {
            "fsspec_kwargs": {
                "s3": {"anon": True},
                "filecache": {"cache_storage": self.cache_storage(job_id)},
            },
            "engine":"h5netcdf"
        }
        return okw


    def my_cycles(self, job_id):
        slices = [slice(st, ed) for st, ed in zip(self.splits[:-1], self.splits[1:])]
        xda = self.xcycles.isel(cycles=slices[job_id])
        cycles_datetime = self.npdate2datetime(xda)
        return cycles_datetime


    def __init__(
        self,
        n_jobs,
        config_filename,
        n_cycles=60, # with two fhr files, about 18 x2 GB cache storage / job
        component="fv3",
        storage_options=None,
        main_cache_path=f"/contrib/Tim.Smith/tmp-replay/1.00-degree",
    ):
        self.n_jobs = n_jobs
        self.n_cycles = n_cycles
        self.config_filename = config_filename
        self.storage_options = storage_options if storage_options is not None else dict()
        self.main_cache_path = main_cache_path

        with open(config_filename, "r") as f:
            config = yaml.safe_load(f)

        name = f"{component.upper()}Dataset" # i.e., FV3Dataset ... maybe an unnecessary generalization at this stage
        self.forecast_hours = config[name]["forecast_hours"]
        self.file_prefixes = config[name]["file_prefixes"]

        assert tuple(self.forecast_hours) == (0, 3)


    def run(self, job_id):
        """Make this essentially a function that can run completely independently of other objects

        Note:
            This could probably be more efficient by creating a list of cycles and passing that to open_dataset,
            rather than going one cycle at a time. However, since we are pulling datasets to local cache and
            I'll be running many jobs concurrently, it could be best to just do it this way.
        """

        localtime = Timer()
        replay = FV3Dataset(path_in=self.cached_path, config_filename=self.config_filename)

        store_coords = False
        for cycles in list(batched(self.my_cycles(job_id), self.n_cycles)):

            localtime.start(f"Reading {str(cycles[0])} - {str(cycles[-1])}")
            xds = replay.open_dataset(list(cycles), **self.ods_kwargs(job_id))

            indices = np.array([list(self.xtime.values).index(t) for t in xds.time.values])
            tslice = slice(indices.min(), indices.max()+1)

            replay.store_dataset(
                    xds,
                    store_coords=store_coords,
                    coords_kwargs={"storage_options": self.storage_options},
                    region={
                        "time": tslice,
                        "pfull": slice(None, None),
                        "grid_yt": slice(None, None),
                        "grid_xt": slice(None, None),
                        },
                    storage_options=self.storage_options,
                    )
            store_coords = False

            # This is a hacky way to clear the cache, since we don't create a filesystem object
            del xds
            if isdir(self.cache_storage(job_id)):
                rmtree(self.cache_storage(job_id), ignore_errors=True)
            localtime.stop()


    def store_container(self):
        """Create an empty container that has the write shape, chunks, and dtype for each variable"""

        localtime = Timer()

        replay = FV3Dataset(path_in=self.cached_path, config_filename=self.config_filename)

        localtime.start("Reading Single Dataset")
        cycle = self.my_cycles(0)[0]
        xds = replay.open_dataset(cycle, **self.ods_kwargs(0))
        xds = xds.reset_coords()
        localtime.stop()

        xds = xds.drop(["ftime", "cftime"])
        data_vars = [x for x in replay.data_vars if x in xds]
        xds = xds[data_vars]

        # Make a container, starting with coordinates
        single = self.remove_time(xds)
        dds = xr.Dataset()
        for key in single.coords:
            dds[key] = xds[key]

        localtime.start("Making container for the dataset")
        dds["time"] = self.xtime
        dds = self.add_time_coords(dds, replay._time2cftime)
        for key in single.data_vars:

            dims = ("time",) + single[key].dims
            chunks = tuple(replay.chunks_out[k] for k in dims)
            shape = (len(dds["time"]),) + single[key].shape

            dds[key] = xr.DataArray(
                    data=darray.zeros(
                        shape=shape,
                        chunks=chunks,
                        dtype=single[key].dtype,
                        ),
                    coords={"time": dds["time"], **{d: single[d] for d in single[key].dims}},
                    dims=dims,
                    attrs=single[key].attrs.copy(),
                )
            print(f"\t ... done with {key}")

        localtime.stop()

        localtime.start("Storing to zarr")
        store = NestedDirectoryStore(path=replay.data_path) if replay.is_nested else replay.data_path
        dds.to_zarr(store, compute=False, storage_options=self.storage_options)
        localtime.stop()

        # This is a hacky way to clear the cache, since we don't create a filesystem object
        del xds
        if isdir(self.cache_storage(0)):
            rmtree(self.cache_storage(0), ignore_errors=True)



    @staticmethod
    def cached_path(dates, forecast_hours, file_prefixes):
        """Note, with simplecache it's not clear where the cached files go, and they
        do not clear until the process is done running (maybe?) which can file up a filesystem easily.
        """

        upper = "filecache::s3://noaa-ufs-gefsv13replay-pds/1deg"
        dates = [dates] if not isinstance(dates, Iterable) else dates

        files = []
        for date in dates:
            this_dir = f"{date.year:04d}/{date.month:02d}/{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}"
            for fp in file_prefixes:
                for fhr in forecast_hours:
                    this_file = join(this_dir, f"{fp}{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}_fhr{fhr:02d}_control")
                    files.append(this_file)
        return [join(upper, this_file) for this_file in files]


    @staticmethod
    def npdate2datetime(npdate):
        if not isinstance(npdate, Iterable):
            return datetime(
                    year=int(npdate.dt.year),
                    month=int(npdate.dt.month),
                    day=int(npdate.dt.day),
                    hour=int(npdate.dt.hour),
                )
        else:
            return [datetime(
                    year=int(t.dt.year),
                    month=int(t.dt.month),
                    day=int(t.dt.day),
                    hour=int(t.dt.hour),
                )
                for t in npdate]


    @staticmethod
    def remove_time(xds):
        single = xds.isel(time=0)
        for key in xds.data_vars:
            if "time" in key:
                del single[key]

        for key in xds.coords:
            if "time" in key:
                del single[key]
        return single

    @staticmethod
    def add_time_coords(xds, time2cftime):
        """add ftime and cftime to container

        This is a bit dirty, passing a static method from another class as a function arg... so it goes.
        """
        ftime = np.array(
                [
                    (np.timedelta64(timedelta(hours=-6)), np.timedelta64(timedelta(hours=-3)))
                        for _ in range(len(xds["time"])//2)
                    ]
                ).flatten()

        xds["ftime"] = xr.DataArray(
                ftime,
                coords=xds["time"].coords,
                dims=xds["time"].dims,
                attrs={
                    "axis": "T",
                    "description": "time passed since forecast initialization",
                    "long_name": "forecast_time",
                    },
                )
        xds["cftime"] = xr.DataArray(
                time2cftime(xds["time"]),
                coords=xds["time"].coords,
                dims=xds["time"].dims,
                attrs={
                    "calendar_type": "JULIAN",
                    "cartesian_axis": "T",
                    "long_name": "time",
                    },
                )
        xds = xds.set_coords(["ftime", "cftime"])
        return xds


class ReplayMoverQuarterDegree(ReplayMover1Degree):

    @property
    def xcycles(self):
        cycles = pd.date_range(start="1994-01-01", end="2023-10-13T06:00:00", freq="6h")
        return xr.DataArray(cycles, coords={"cycles": cycles}, dims="cycles")


    @property
    def xtime(self):
        time = pd.date_range(start="1994-01-01", end="2023-10-13T09:00:00", freq="3h")
        iau_time = time - timedelta(hours=6)
        return xr.DataArray(iau_time, coords={"time": iau_time}, dims="time", attrs={"long_name": "time", "axis": "T"})

    def __init__(
        self,
        n_jobs,
        config_filename,
        n_cycles=4, # with two fhr files, about 16 x2 GB cache storage / job
        component="fv3",
        storage_options=None,
        main_cache_path=f"/contrib/Tim.Smith/tmp-replay/0.25-degree",
    ):
        super().__init__(
            n_jobs=n_jobs,
            n_cycles=n_cycles,
            config_filename=config_filename,
            component=component,
            storage_options=storage_options,
            main_cache_path=main_cache_path,
        )

    @staticmethod
    def cached_path(dates, forecast_hours, file_prefixes):
        """Note, with simplecache it's not clear where the cached files go, and they
        do not clear until the process is done running (maybe?) which can file up a filesystem easily.
        """

        upper = "filecache::s3://noaa-ufs-gefsv13replay-pds"
        dates = [dates] if not isinstance(dates, Iterable) else dates

        files = []
        for date in dates:
            this_dir = f"{date.year:04d}/{date.month:02d}/{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}"
            for fp in file_prefixes:
                for fhr in forecast_hours:
                    this_file = join(this_dir, f"{fp}{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}_fhr{fhr:02d}_control")
                    files.append(this_file)
        return [join(upper, this_file) for this_file in files]


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch
