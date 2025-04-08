import os
import shutil
import itertools
import logging

from math import ceil

import xarray as xr
import dask.array

from .mpi import MPITopology, _has_mpi

logger = logging.getLogger("ufs2arco")

class DataMover():
    """Move data, using the concept of a data "sample" to define how much data is stored to zarr at once.
    A data sample is defined by :attr:`sample_dims`. For example, if a dataset has dimensions
    ``("t0", "fhr", "member", "pressure", "latitude", "longitude")``
    then setting ``sample_dims = ("t0", "fhr", "member")``
    would loop through the data using those indices, where each sample has the full
    ``("pressure", "latitude", "longitude")`` array for the given values of ``("t0", "fhr", "member")``.

    Note:
        * For now only works with GEFSDataset
        * This is the same as loop through the data storing one sample at a time,
          except that it stores ``batch_size`` samples in a hard disk cache.
          It clears the cache after every batch.
        * Multithreading for simultaneous reading/writing could be implemented, but hasn't seemed necessary.
          Also it would double (or more) the cache requirements.
    """
    counter = 0
    data_counter = 0

    stop_event = None
    executor = None
    futures = None

    def __init__(
        self,
        source,
        target,
        batch_size,
        transformer=None,
        start=0,
        cache_dir=".",
    ):

        self.source = source
        self.target = target
        self.transformer = transformer if transformer is not None else lambda xds: xds
        self.batch_size = batch_size
        self.counter = start
        self.data_counter = start
        self.outer_cache_dir = cache_dir

        # construct the sample indices
        # e.g. {"t0": [date1, date2], "fhrs": [0, 6], "member": [0, 1, 2]}
        all_sample_iterations = {
            key: getattr(source, key)
            for key in self.source.sample_dims
        }
        self.sample_indices = list()
        for combo in itertools.product(*all_sample_iterations.values()):
            self.sample_indices.append(dict(zip(self.source.sample_dims, combo)))

        self.restart(idx=start)

    @property
    def name(self):
        return str(type(self).__name__)


    def __len__(self) -> int:
        n_samples = len(self.sample_indices)
        n_batches = ceil(n_samples / self.batch_size)
        return n_batches

    def __iter__(self):
        self.counter = 0
        self.restart()
        return self

    def __next__(self):
        if self.counter < len(self):
            data = self.get_data()
            self.counter += 1
            return data
        else:
            logger.debug(f"{self.name}.__next__: counter > len(self)")
            raise StopIteration


    def get_cache_dir(self, batch_idx):
        return f"{self.outer_cache_dir}/{self.name.lower()}-cache/{batch_idx}"


    def get_batch_indices(self, batch_idx):
        st = batch_idx * self.batch_size
        ed = st + self.batch_size
        return self.sample_indices[st:ed]


    def _next_data(self):
        logger.debug(f"{self.name}._next_data[{self.data_counter}]")
        if self.data_counter < len(self):
            batch_indices = self.get_batch_indices(self.data_counter)
            cache_dir = self.get_cache_dir(self.data_counter)
            if len(batch_indices) > 0:
                dlist = []

                for these_dims in batch_indices:
                    fds = self.source.open_sample_dataset(
                        dims=these_dims,
                        open_static_vars=self.target.always_open_static_vars,
                        cache_dir=cache_dir,
                    )
                    if len(fds) > 0:
                        fds = self.transformer(fds)
                        fds = self.target.apply_transforms_to_sample(fds)
                    dlist.append(fds)
                xds = xr.merge(dlist)
                return xds

            else:
                return None
        else:
            logger.debug(f"{self.name}._next_data: data_counter > len(self)")
            raise StopIteration


    def get_data(self):
        """Pull a batch of data from the queue"""
        logger.debug(f"{self.name}.get_data")
        data = self._next_data()
        self.data_counter += 1
        return data


    def clear_cache(self, batch_idx):
        cache_dir = self.get_cache_dir(batch_idx)
        if os.path.isdir(cache_dir):
            logger.debug(f"{self.name}: clearing {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)

    def restart(self, idx=0):
        """Restart the :attr:`data_counter` to get ready for the pass through the data

        Args:
            idx (int, optional): index to restart to
        """
        logger.debug(f"{self.name}.restart: idx = {idx}")
        self.data_counter = idx


    def create_container(self, **kwargs) -> None:
        """
        Create a Zarr container to store the dataset.

        Args:
            **kwargs: Additional arguments passed to `xr.Dataset.to_zarr`.

        Logs:
            The created container is stored at `self.target.store_path`.
        """

        # open a minimal dataset
        first_sample_dim_args = {
            key: getattr(self.source, key)[0]
            for key in self.source.sample_dims
        }
        xds = self.source.open_sample_dataset(
            dims=first_sample_dim_args,
            open_static_vars=True,
            cache_dir=self.get_cache_dir("container"),
        )

        # perform any transformations like regridding
        xds = self.transformer(xds)

        # transform it to target space
        xds = self.target.apply_transforms_to_sample(xds)

        # create container
        # start with the dimensions we haven't read yet (sample_dims)
        nds = xr.Dataset(attrs=xds.attrs.copy())
        for key in self.target.renamed_sample_dims:
            array = getattr(self.target, key)
            nds[key] = xr.DataArray(
                array,
                coords={key: array},
                dims=key,
                attrs=xds[key].attrs.copy(),
            )

        # these will be the the verical dim + horizontal_dims, we read these in each sample
        for key in xds.dims:
            if key not in self.target.renamed_sample_dims:
                nds[key] = xds[key].copy()

        # manage coordinates
        # first we have to pass all existing coordinates,
        # then let target manage them
        for key in xds.coords:
            if key not in nds:
                nds = nds.assign_coords({key: xds[key].copy()})
        nds = self.target.manage_coords(nds)

        # create empty data arrays
        for varname in xds.data_vars:
            dims = xds[varname].dims
            shape = tuple(len(nds[key]) for key in dims)
            chunks = {list(dims).index(key): self.target.chunks[key] for key in dims}
            nds[varname] = xr.DataArray(
                data=dask.array.zeros(
                    shape=shape,
                    chunks=chunks,
                    dtype=xds[varname].dtype,
                ),
                dims=dims,
                attrs=xds[varname].attrs.copy(),
            )

        nds.to_zarr(self.target.store_path, compute=False, **kwargs)
        logger.info(f"{self.name}.create_container: stored container at {self.target.store_path}\n{nds}\n")

        self.clear_cache("container")


    def find_my_region(self, xds):
        """Given a dataset, that's assumed to be a subset of the initial dataset,
        find the logical index values where this should be stored in the final zarr store

        Args:
            xds (xr.Dataset): with a subset of the data (i.e., a couple of initial conditions)

        Returns:
            region (dict): indicating the zarr region to store in, based on the initial condition indices
        """
        region = {k: slice(None, None) for k in xds.dims}
        for key in self.target.renamed_sample_dims:
            full_array = getattr(self.target, key) # e.g. all of the initial conditions
            batch_indices = [list(full_array).index(value) for value in xds[key].values]
            region[key] = slice(batch_indices[0], batch_indices[-1]+1)
        return region



class MPIDataMover(DataMover):
    """Use MPI to scale up the data moving. Note that this pulls in one data sample per MPI proces
    (i.e., :attr:`data_per_process` always = 1), and uses the MPI size (i.e., number of processes)
    as the :attr:`batch_size`.

    So, this also requires ``num_mpi_processes`` = :attr:`batch_size` * `sample_size_in_bytes` disk space
    for caching.
    """
    def __init__(
        self,
        source,
        target,
        mpi_topo,
        transformer=None,
        start=0,
        cache_dir=".",
    ):
        assert _has_mpi, f"{self.name}.__init__: Unable to import mpi4py, cannot use this class"

        self.topo = mpi_topo
        batch_size = self.topo.size
        self.data_per_process = 1
        self.local_batch_index = self.topo.rank*self.data_per_process
        super().__init__(
            source=source,
            target=target,
            batch_size=batch_size,
            transformer=transformer,
            start=start,
            cache_dir=cache_dir,
        )
        logger.info(str(self))

    def __str__(self):
        title = f"Mover: {self.name}"
        underline = "".join(["-" for _ in range(len(title))])
        msg = f"\n{title}\n{underline}\n"

        for key in ["local_batch_index", "data_per_process", "batch_size"]:
            msg += f"{key:<18s}: {getattr(self, key):02d}\n"

        msg += f"{'Total Samples':<18s}: {len(self.sample_indices)}\n"
        msg += f"{'Total Batches':<18s}: {len(self)}\n"
        msg += f"{'sample_dims':<18s}: {self.source.sample_dims}\n"
        return msg

    def get_cache_dir(self, batch_idx):
        return f"{self.outer_cache_dir}/{self.name.lower()}-cache/{self.topo.rank}/{batch_idx}"

    def get_batch_indices(self, batch_idx):
        st = (batch_idx * self.batch_size) + self.local_batch_index
        ed = st + self.data_per_process
        return self.sample_indices[st:ed]

    def create_container(self, **kwargs) -> None:
        if self.topo.is_root:
            super().create_container(**kwargs)
        self.topo.barrier()
