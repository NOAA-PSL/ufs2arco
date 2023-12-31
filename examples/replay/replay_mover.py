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

from ufs2arco import FV3Dataset, Timer

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

    forecast_hours = None
    file_prefixes = None

    @property
    def xcycles(self):
        """These are the DA cycle timestamps, which are every 6 hours. There is one s3 directory per cycle for replay."""
        cycles = pd.date_range(start="1994-01-01", end="1999-06-13T06:00:00", freq="6h")
        return xr.DataArray(cycles, coords={"cycles": cycles}, dims="cycles")


    @property
    def xtime(self):
        """These are the time stamps of the resulting dataset, assuming we are grabbing fhr00 and fhr03.

        This was created by processing a few DA cycles with the desired forecast hours, and figuring out
        operations were needed to map from a list of all DA cycles to the resulting ``"time"``
        array in the final dataset.
        """
        time = pd.date_range(start="1994-01-01", end="1999-06-13T09:00:00", freq="3h")
        iau_time = time - timedelta(hours=6)
        return xr.DataArray(iau_time, coords={"time": iau_time}, dims="time", attrs={"long_name": "time", "axis": "T"})


    @property
    def splits(self):
        """The indices used to split all cycles across :attr:`n_jobs`"""
        return [int(x) for x in np.linspace(0, len(self.xcycles), self.n_jobs+1)]

    def cache_storage(self, job_id):
        """Location to store the s3 data, before subsetting, rechunking, and pushing to GCS

        Args:
            job_id (int): the slurm job id, determines cache storage location

        Returns:
            cache_storage (str): location to store s3 netcdf files
        """
        return join(self.main_cache_path, str(job_id))

    def ods_kwargs(self, job_id):
        """These are passed to xarray.open_dataset to read from s3 and store the file in cache

        Args:
            job_id (int): the slurm job id, determines cache storage location

        Returns:
            xarray_open_dataset_kwargs (dict): job_id determines cache_storage location
        """
        okw = {
            "fsspec_kwargs": {
                "s3": {"anon": True},
                "filecache": {"cache_storage": self.cache_storage(job_id)},
            },
            "engine":"h5netcdf"
        }
        return okw


    def my_cycles(self, job_id):
        """The cycle timestamps for this job

        Args:
            job_id (int): the slurm job id, determines cache storage location

        Returns:
            cycles_datetime (List[datetime]): with the cycle numbers to be processed by this slurm job
        """
        slices = [slice(st, ed) for st, ed in zip(self.splits[:-1], self.splits[1:])]
        xda = self.xcycles.isel(cycles=slices[job_id])
        cycles_datetime = self.npdate2datetime(xda)
        return cycles_datetime


    def __init__(
        self,
        n_jobs,
        config_filename,
        component="fv3",
        storage_options=None,
        main_cache_path=f"/contrib/Tim.Smith/tmp-replay/1.00-degree",
    ):
        self.n_jobs = n_jobs
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
        """This pulls data and stores it to the desired storage location.
        This is essentially a function that can run completely independently of other objects

        Note:
            This expects :meth:`.store_container` to have been run first

        Args:
            job_id (int): the slurm job id, determines cache storage location
        """

        walltime = Timer()
        localtime = Timer()
        replay = FV3Dataset(path_in=self.cached_path, config_filename=self.config_filename)

        store_coords = False # we don't need a separate coords dataset for FV3
        for cycle in self.my_cycles(job_id):

            localtime.start(f"Reading {str(cycle)}")

            try:
                self.move_single_dataset(cycle)
            except Exception as e:
                logging.exception(e)
                print(f"ReplayMover.run({job_id}): Failed to store {str(cycle)}")
                pass
            localtime.stop()

        walltime.stop("Total Walltime")

    def move_single_dataset(self, cycle):
        """Store a single cycle to zarr"""

        xds = replay.open_dataset(cycle, **self.ods_kwargs(job_id))

        index = list(self.xtime.values).index(xds.time.values[0])
        tslice = slice(index, index+2)

        replay.store_dataset(
            xds,
            region={
                "time": tslice,
                "pfull": slice(None, None),
                "grid_yt": slice(None, None),
                "grid_xt": slice(None, None),
                },
            storage_options=self.storage_options,
        )
        # This is a hacky way to clear the cache, since we don't create a filesystem object
        del xds
        if isdir(self.cache_storage(job_id)):
            rmtree(self.cache_storage(job_id), ignore_errors=True)


    def store_container(self):
        """Create an empty container that has the write shape, chunks, and dtype for each variable
        """

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
        """This is passed to :class:`FV3Dataset`, and it generates the paths to read from for the given inputs

        Note:
            With simplecache it's not clear where the cached files go, and they
            do not clear until the process is done running (maybe?) which can file up a filesystem easily.
            So we use filecache instead.

        Args:
            dates (Iterable[datetime]): with the DA cycles to read from
            forecast_hours (List[int]): with the forecast hours to grab ... note here we assume [0, 3] ... but don't enforce it here
            file_prefixes (List[str]): e.g. ["sfg_", "bfg_"]

        Returns:
            list_of_paths (List[str]): see example

        Example:
            >>> mover = ReplayMover( ... )
            >>> mover.cached_path(
                    dates=[datetime(1994,1,1,0), datetime(1994,1,1,6)],
                    forecast_hours=[0, 3],
                    file_prefixes=["sfg_", "bfg_"],
                )
                ["filecache::s3://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/1deg/1994/01/1994010100/sfg_1994010100_fhr00_control,
                "filecache::s3://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/1deg/1994/01/1994010100/sfg_1994010100_fhr03_control
                "filecache::s3://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/1deg/1994/01/1994010100/bfg_1994010100_fhr00_control,
                "filecache::s3://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/1deg/1994/01/1994010100/bfg_1994010100_fhr03_control
                "filecache::s3://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/1deg/1994/01/1994010106/sfg_1994010106_fhr00_control,
                "filecache::s3://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/1deg/1994/01/1994010106/sfg_1994010106_fhr03_control
                "filecache::s3://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/1deg/1994/01/1994010106/bfg_1994010106_fhr00_control,
                "filecache::s3://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/1deg/1994/01/1994010106/bfg_1994010106_fhr03_control]
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
        """Convert numpy.datetime64 to datetime.datetime"""
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
        """Remove time from dataset"""
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
        """These are the DA cycle timestamps, which are every 6 hours. There is one s3 directory per cycle for replay."""
        cycles = pd.date_range(start="1994-01-01", end="2023-10-13T06:00:00", freq="6h")
        return xr.DataArray(cycles, coords={"cycles": cycles}, dims="cycles")


    @property
    def xtime(self):
        """These are the time stamps of the resulting dataset, assuming we are grabbing fhr00 and fhr03"""
        time = pd.date_range(start="1994-01-01", end="2023-10-13T09:00:00", freq="3h")
        iau_time = time - timedelta(hours=6)
        return xr.DataArray(iau_time, coords={"time": iau_time}, dims="time", attrs={"long_name": "time", "axis": "T"})

    def __init__(
        self,
        n_jobs,
        config_filename,
        component="fv3",
        storage_options=None,
        main_cache_path=f"/contrib/Tim.Smith/tmp-replay/0.25-degree",
    ):
        super().__init__(
            n_jobs=n_jobs,
            config_filename=config_filename,
            component=component,
            storage_options=storage_options,
            main_cache_path=main_cache_path,
        )

    @staticmethod
    def cached_path(dates, forecast_hours, file_prefixes):
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
