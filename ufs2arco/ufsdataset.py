from os.path import join
from collections.abc import Iterable
from typing import Dict, List, Callable
import fsspec
import yaml
import warnings

import numpy as np
import xarray as xr
from zarr import NestedDirectoryStore

from datetime import datetime, timedelta
from cftime import DatetimeJulian

from .utils import batched

class UFSDataset:
    """Open and store a UFS generated NetCDF dataset to zarr from a single model component (FV3, MOM6, CICE6). The two main methods that are useful are :meth:`open_dataset` and :meth:`store_dataset`.

    Note:
        The ``path_in`` argument on __init__ probably needs some attention, especially in relation to the ``file_prefixes`` option. This should be addressed once we start thinking about datasets other than replay.

    Required Fields in Config:
        path_out (str): the outermost directory to store the dataset
        forecast_hours (list of int): with the forecast hours to save
        file_prefixes (list of str): with the filename prefixes to read, e.g. ["sfg", "bfg"] to read all files starting with this prefix at this cycle, for all forecast_hours

    Optional Fields in Config:
        coords (list of str): containing static coordinate variables to store only one time
        coords_path_out (str): path to store coordinates, if not provided then store under 'coordinates' directory in data_path
        data_vars (list of str): containing variables that evolve in time to be stored if not provided, all variables will be stored
        chunks_in, chunks_out (dict): containing chunksizes for each dimension

    Args:
        path_in (callable): map the following arguments to a path (str):
        config_filename (str): to yaml file containing the overall configuration
        is_nested (bool, optional): if True write a :class:`NestedDirectoryStore`, otherwise write Zarr's default directory structure

    Sets Attributes:
        path_out (str): the outermost directory to store the dataset
        forecast_hours (list of int): with the forecast hours to save
        file_prefixes (list of str): with the filename prefixes inside of each cycle's directory,
        coords, data_vars (list): with variable names of coordinates and data variables
        chunks_in, chunks_out (dict): specifying how to chunk the data when reading and writing
        config (dict): with the configuration provided by the file
        is_nested (bool): whether or not to write with :class:`NestedDirectoryStore`
    """

    @property
    def data_path(self):
        """Where to write forecast data variables to"""
        return self._join(self.path_out, self.zarr_name)

    @property
    def coords_path(self) -> str:
        """Where to write static coordinates to"""
        if self.coords_path_out is None:
            return self._join(self.path_out, "coordinates", self.zarr_name)
        else:
            return self._join(self.coords_path_out, self.zarr_name)

    @property
    def default_open_dataset_kwargs(self) -> Dict:
        kw = {
            "parallel": True,
            "chunks": self.chunks_in,
            "decode_times": True,
            "preprocess": self._preprocess,
        }
        return kw

    def __init__(self, path_in: Callable, config_filename: str, is_nested: bool = False) -> None:
        super(UFSDataset, self).__init__()
        name = self.__class__.__name__  # e.g., FV3Dataset, MOMDataset

        # create and initialze instance variable for class attributes
        self.path_in: Callable = path_in
        self.is_nested: bool = is_nested
        self.path_out: str = ""
        self.forecast_hours: List[int] = []
        self.file_prefixes: List[str] = []

        self.chunks_in: Dict = {}
        self.chunks_out: Dict = {}
        self.coords: List[str] = []
        self.data_vars: List[str] = []
        self.zarr_name: str = ""
        self.coords_path_out = None

        with open(config_filename, "r") as f:
            contents = yaml.safe_load(f)
            self.config = contents[name]

        # look for these requited inputs
        for key in ["path_out", "forecast_hours", "file_prefixes"]:
            try:
                setattr(self, key, self.config[key])
            except KeyError:
                raise KeyError(f"{name}.__init__: Could not find {key} in {config_filename}, but this is required")

        # look for these optional inputs
        for key in ["chunks_in", "chunks_out", "coords", "data_vars", "coords_path_out"]:
            if key in self.config:
                setattr(self, key, self.config[key])
            else:
                print(f"{name}.__init__: Could not find {key} in {config_filename}, using default.")

        # warn user about not finding coords
        if len(self.coords) == 0:
            warnings.warn(
                f"{name}.__init__: Could not find 'coords' in {config_filename}, will not store coordinate data"
            )

        if len(self.data_vars) == 0:
            warnings.warn(
                f"{name}.__init__: Could not find 'data_vars' in {config_filename}, will store all data variables"
            )

        # check that file_prefixes is a list
        self.file_prefixes = [self.file_prefixes] if isinstance(self.file_prefixes, str) else self.file_prefixes

    def open_dataset(self, cycles: datetime, fsspec_kwargs=None, **kwargs):
        """Read data from specified DA cycles

        Args:
            cycles (datetime.datetime or List[datetime.datetime]): datetime object(s) giving initial time for each DA cycles
            fsspec_kwargs (dict, optional): optional arguments passed to :func:`fsspec.open_files`
            **kwargs (dict, optional): optional arguments passed to :func:`xarray.open_mfdataset`, in addition to
                the ones provided by :attr:`default_open_dataset_kwargs`

        Returns:
            xds (xarray.Dataset): with output from a single model component (e.g., FV3)
                and a single forecast window, with all output times in
                that window merged into one dataset
        """

        kw = self.default_open_dataset_kwargs.copy()
        kw.update(kwargs)

        fnames = self.path_in(cycles, self.forecast_hours, self.file_prefixes)

        # Maybe there's a more elegant way to handle this, but with local files, fsspec closes them
        # before dask reads them...
        if fsspec_kwargs is None:
            xds = xr.open_mfdataset(fnames, **kw)
        else:
            with fsspec.open_files(fnames, **fsspec_kwargs) as files:
                # This assumes netcdf4, which I think is a safe fallback for UFS data
                engine = kw.pop("engine", "netcdf4")
                if engine == "netcdf4":
                    xds = xr.open_mfdataset([this_file.name for this_file in files], **kw)
                else:
                    xds = xr.open_mfdataset(files, **kw)
        return xds

    def chunk(self, xds):
        """Using the yaml-provided or default chunking scheme, chunk all arrays in this dataset

        Note:
            This should probably be replaced with rechunker https://rechunker.readthedocs.io/en/latest/

        Args:
            xds (xarray.Dataset): as provided by :meth:`open_dataset`

        Returns:
            xds (xarray.Dataset): rechunked as specified
        """

        chunks = self.chunks_out.copy()
        for key in self.chunks_out.keys():
            if key not in xds.dims:
                chunks.pop(key)

        xds = xds.transpose(*list(chunks.keys()))
        return xds.chunk(chunks)

    def store_dataset(self,
        xds: xr.Dataset,
        store_coords: bool = False,
        coords_kwargs=None,
        **kwargs) -> None:
        """Open all netcdf files for this model component and at this DA window, store
        coordinates one time only, select data based on
        desired forecast hour, then store it.

        Args:
            xds (xarray.Dataset): as provided by :meth:`open_dataset`
            store_coords (bool, optional): if True, store coordinates in separate directory
            coords_kwargs (dict, optional): passed to :func:`xarray.to_zarr` via :meth:`_store_coordinates`
            kwargs (dict): optional arguments passed to :func:`xarray.to_zarr` via :meth:`_store_data_vars`
        """

        xds = xds.reset_coords()

        # need to store coordinates dataset only one time
        if store_coords:
            coords = [x for x in self.coords if x in xds]
            cds = xds[coords].set_coords(coords)
            if "member" in cds:
                cds = cds.isel(member=0).drop("member")
            ckw = dict() if coords_kwargs is None else coords_kwargs
            self._store_coordinates(cds, **ckw)

        # now data variables at this cycle
        # make various time variables as coordinates
        xds = xds.set_coords(["time", "cftime", "ftime"])
        if len(self.data_vars) > 0:
            data_vars = [x for x in self.data_vars if x in xds]
            xds = xds[data_vars]

        self._store_data_vars(xds, **kwargs)

    def _store_coordinates(self, cds: xr.Dataset, **kwargs) -> None:
        """Store the static coordinate information to zarr

        Args:
            cds (xarray.Dataset): with only the static coordinate information
        """

        try:
            assert len(cds.data_vars) == 0
        except AssertionError:
            msg = "UFSDataset._store_coordinates: "+\
                f"We should not have any data variables in this dataset, but we found some."+\
                f"\n{cds.data_vars}"
            raise AttributeError(msg)

        # these don't need to be chunked, coordinates are opened in memory
        store = NestedDirectoryStore(path=self.coords_path) if self.is_nested else self.coords_path
        cds.to_zarr(store, **kwargs)
        print(f"Stored coordinate dataset at {self.coords_path}")

    def _store_data_vars(self, xds: xr.Dataset, **kwargs) -> None:
        """Store the data variables

        Args:
            xds (xarray.Dataset): the big dataset with all desired data variables, for this model component
                and at this particular DA window
        """

        xds = self.chunk(xds)

        store = NestedDirectoryStore(path=self.data_path) if self.is_nested else self.data_path
        xds.to_zarr(store, **kwargs)
        print(f"Stored dataset at {self.data_path}")

    @staticmethod
    def _preprocess(xds):
        """Used to remove a redundant surface pressure found in both physics and dynamics FV3 files,
        which are slightly different and so cause a conflict. This method is not used on its own, but
        given as an option to :func:`xarray.open_mfdataset`

        Args:
            xds (xarray.Dataset): A single netcdf file from the background forecast

        Returns:
            xds (xarray.Dataset): with ``pressfc`` variable removed if this is from FV3 dynamics output
        """
        # We can't rely on xds.encoding when reading from s3, so have to infer if this is dynamics
        # vs physics dataset by the other fields that exist in the dataset
        dyn_vars = ["tmp", "ugrd", "vgrd", "spfh", "o3mr"]
        if "pressfc" in xds.data_vars and any(v in xds.data_vars for v in dyn_vars):
            del xds["pressfc"]
        return xds

    @staticmethod
    def _cftime2time(cftime):
        """Convert cftime array to numpy.datetime64

        Args:
            cftime (xarray.DataArray): with DatetimeJulian objects

        Returns:
            xtime (xarray.DataArray): with numpy.datetime64 objects
        """
        time = np.array(
            [
                np.datetime64(
                    datetime(
                        int(t.dt.year),
                        int(t.dt.month),
                        int(t.dt.day),
                        int(t.dt.hour),
                        int(t.dt.minute),
                        int(t.dt.second),
                    )
                )
                for t in cftime
            ]
        )
        xtime = xr.DataArray(
            time,
            coords=cftime.coords,
            dims=cftime.dims,
            attrs={
                "long_name": "time",
                "axis": "T",
            },
        )
        return xtime

    @staticmethod
    def _time2cftime(time):
        """Convert numpy.datetime64 array to cftime

        Args:
            time (xarray.DataArray): with numpy.datetime64 objects

        Returns:
            xcftime (xarray.DataArray): with DatetimeJulian objects
        """
        cftime = np.array(
            [
                DatetimeJulian(
                    int(t.dt.year),
                    int(t.dt.month),
                    int(t.dt.day),
                    int(t.dt.hour),
                    int(t.dt.minute),
                    int(t.dt.second),
                    has_year_zero=False,
                )
                for t in time
            ]
        )
        xcftime = xr.DataArray(
            cftime,
            coords=time.coords,
            dims=time.dims,
            attrs={
                "long_name": "time",
                "axis": "T",
                "calendar_type": "JULIAN",
            },
        )
        return xcftime

    @staticmethod
    def _time2ftime(time, cycles):
        """Compute the ftime (forecast_time) array, indicating the hours since
        initialization for each timestamp

        Args:
            time (xr.DataArray): with numpy.datetime64 objects
            cycles (datetime or List[datetime]): with the DA cycle(s) to grab

        Returns:
            xftime (xr.DataArray): forecast_time
        """
        cycles = [cycles] if not isinstance(cycles, Iterable) else cycles
        n_output_per_cycle = len(time) // len(cycles)
        time_batches = list(batched(time.values, n_output_per_cycle))
        ftime = np.array([
            these_times - np.datetime64(this_cycle) for these_times, this_cycle in zip(time_batches, cycles)
        ]).flatten()
        xftime = xr.DataArray(
            ftime,
            coords=time.coords,
            dims=time.dims,
            attrs={
                "long_name": "forecast_time",
                "description": f"time passed since forecast initialization",
                "axis": "T",
            },
        )
        return xftime

    @staticmethod
    def _join(a, *p):
        """System independent join operation"""
        clouds = ("gcs://", "s3://", "https://")
        if any(x in a for x in clouds) or any(any(x in this_path for x in clouds) for this_path in p):
            try:
                assert isinstance(a, str) and all(isinstance(this_path, str) for this_path in p)
            except:
                raise TypeError(f"For cloud storage, paths need to be strings.")

            path = a
            join_char = "/" if a[-1] != "/" else ""
            for this_path in p:
                path += join_char
                path += this_path
                join_char = "/" if this_path[-1] != "/" else ""

            return path

        else:
            return join(a, *p)
