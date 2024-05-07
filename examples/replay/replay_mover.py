from os.path import join, isdir
from datetime import datetime, timedelta
import logging
import yaml
from shutil import rmtree
import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as darray
from zarr import NestedDirectoryStore

from ufs2arco import (
    FV3Dataset, 
    MOM6Dataset, 
    CICE6Dataset, 
    Timer, 
    MOM6Regridder,
    CICE6Regridder
)

class ReplayMover1Degree():

    n_jobs = None

    forecast_hours = None
    file_prefixes = None

    @property
    def xcycles(self):
        return xr.DataArray(self.cycles, coords={"cycles": self.cycles}, dims="cycles")


    @property
    def xtime(self):
        """These are the time stamps of the resulting dataset.

        This was created by processing a few DA cycles with the desired forecast hours, and figuring out
        operations were needed to map from a list of all DA cycles to the resulting ``"time"``
        array in the final dataset.
        """
        return xr.DataArray(self.time, coords={"time": self.time}, dims="time", attrs={"long_name": "time", "axis": "T"})


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
        self.component = component
        self.storage_options = storage_options if storage_options is not None else dict()
        self.main_cache_path = main_cache_path

        with open(config_filename, "r") as f:
            config = yaml.safe_load(f)

        name = f"{component.upper()}Dataset" # i.e., FV3Dataset ... maybe an unnecessary generalization at this stage
        self.forecast_hours = config[name]["forecast_hours"]
        self.file_prefixes = config[name]["file_prefixes"]
        self.cycles = pd.date_range(**config[name]["cycles"])
        self.time = pd.date_range(**config[name]["time"])

        self.Regridder = None
        if component.lower() == "fv3":
            self.Dataset = FV3Dataset
            self.cached_path = self.fv3_path
            if "regrid" in self.config.keys():
                raise NotImplementedError
        elif component.lower() == "mom6":
            self.Dataset = MOM6Dataset
            self.cached_path = self.mom6_path
            self.Regridder = MOM6Regridder
        elif component.lower() == "cice6":
            self.Dataset = CICE6Dataset
            self.cached_path = self.cice6_path
            self.Regridder = CICE6Regridder

        # for move_single_dataset, we have to figure out how many resulting timestamps we have
        # within a single DA cycle
        try:
            assert "freq" in config[name]["cycles"].keys()
            assert "freq" in config[name]["time"].keys()

        except:
            raise KeyError("ReplayMover.__init__: we need 'freq' inside the config 'cycles' and 'time' sections")
        delta_t_cycle = pd.Timedelta(config[name]["cycles"]["freq"])
        delta_t_time  = pd.Timedelta(config[name]["time"]["freq"])
        self.n_steps_per_cycle = delta_t_cycle // delta_t_time


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
        walltime.start()

        store_coords = False # we don't need a separate coords dataset for FV3
        for cycle in self.my_cycles(job_id):

            localtime.start(f"Reading {str(cycle)}")

            try:
                self.move_single_dataset(job_id, cycle)
            except Exception as e:
                logging.exception(e)
                print(f"ReplayMover.run({job_id}): Failed to store {str(cycle)}")
                pass
            localtime.stop()

        walltime.stop("Total Walltime")
    
    def regrid_to_gaussian(
            self,
            xds: xr.Dataset,
            gaussian_grid_path: str,
    ) -> xr.Dataset:
        """Regrid (if needed) - will be used of ocean/sea ice data"""
        gaussian_grid = xr.open_zarr(
            gaussian_grid_path,
            storage_options={"token": "anon"},
            )
        lons = gaussian_grid['grid_xt'].values
        lats = gaussian_grid['grid_yt'].values
        rg = self.Regridder(
            lats1d_out=lats, 
            lons1d_out=lons,
            ds_in=xds, 
            config_filename=self.config_filename,
            )
        xds = rg.regrid(xds)
        
        return xds
        

    def move_single_dataset(self, job_id, cycle):
        """Store a single cycle to zarr"""

        replay = self.Dataset(path_in=self.cached_path, config_filename=self.config_filename)
        xds = replay.open_dataset(cycle, **self.ods_kwargs(job_id))

        if "regrid" in replay.config.keys():
            xds = self.regrid_to_gaussian(
                xds=xds,
                gaussian_grid_path=replay.config['gaussian_grid'],
                )

        index = list(self.xtime.values).index(xds.time.values[0])
        tslice = slice(index, index+self.n_steps_per_cycle)

        # subset the variables here in order to remove extraneous dimensions
        if len(replay.data_vars)>0:
            data_vars = [x for x in replay.data_vars if x in xds]
            xds = xds[data_vars]

        spatial_region = {k: slice(None, None) for k in xds.dims if k != "time"}
        region = {
            "time": tslice,
            **spatial_region,
        }
        region = {k : v for k,v in region.items() if k in xds.dims}

        replay.store_dataset(
            xds,
            region=region,
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

        replay = self.Dataset(path_in=self.cached_path, config_filename=self.config_filename)

        localtime.start("Reading Single Dataset")
        cycle = self.my_cycles(0)[0]
        xds = replay.open_dataset(cycle, **self.ods_kwargs(0))

        if "regrid" in replay.config.keys():
            xds = self.regrid_to_gaussian(
                xds=xds,
                gaussian_grid_path=replay.config['gaussian_grid'],
                )

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
    def fv3_path(dates, forecast_hours, file_prefixes):
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
    def mom6_path(cycles, forecast_hours, file_prefixes):
        """This is passed to :class:`MOM6Dataset`, and it generates the paths to read from for the given inputs

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
                    forecast_hours=[0],
                    file_prefixes=["ocn_"],
                )
				['filecache::s3://noaa-ufs-gefsv13replay-pds/1deg/1994/01/1994010100/ocn_1994_01_01_00.nc',
                'filecache::s3://noaa-ufs-gefsv13replay-pds/1deg/1994/01/1994010106/ocn_1994_01_01_06.nc']
        """
        upper = "filecache::s3://noaa-ufs-gefsv13replay-pds/1deg"
        cycles = [cycles] if not isinstance(cycles, list) else cycles

        files = []
        for cycle in cycles:
            this_dir = f"{cycle.year:04d}/{cycle.month:02d}/{cycle.year:04d}{cycle.month:02d}{cycle.day:02d}{cycle.hour:02d}"

            for fp in file_prefixes:
                for fhr in forecast_hours:
                    this_date = cycle+timedelta(hours=fhr)
                    this_file = f"{this_dir}/{fp}{this_date.year:04d}_{this_date.month:02d}_{this_date.day:02d}_{this_date.hour:02d}.nc"
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

    def add_time_coords(self, xds, time2cftime):
        """add ftime and cftime to container

        This is a bit dirty, passing a static method from another class as a function arg... so it goes.
        """

        iau_offset = -6 if self.component == "fv3" else 0
        repeater = tuple(np.timedelta64(timedelta(hours=fhr-iau_offset)) for fhr in self.forecast_hours)
        ftime = np.array(
            [repeater for _ in range(len(xds["time"])//len(repeater))]
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
    def fv3_path(dates, forecast_hours, file_prefixes):
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

    @staticmethod
    def mom6_path(cycles, forecast_hours, file_prefixes):
        upper = "filecache::s3://noaa-ufs-gefsv13replay-pds"
        cycles = [cycles] if not isinstance(cycles, list) else cycles

        files = []
        for cycle in cycles:
            this_dir = f"{cycle.year:04d}/{cycle.month:02d}/{cycle.year:04d}{cycle.month:02d}{cycle.day:02d}{cycle.hour:02d}"
            for fp in file_prefixes:
                for fhr in forecast_hours:
                    this_date = cycle+timedelta(hours=fhr)
                    this_file = f"{this_dir}/{fp}{this_date.year:04d}_{this_date.month:02d}_{this_date.day:02d}_{this_date.hour:02d}.nc"
                    files.append(this_file)
        return [join(upper, this_file) for this_file in files]
