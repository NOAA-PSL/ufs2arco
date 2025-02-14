from math import ceil
import numpy as np

import itertools
import logging
import threading
import concurrent
import queue

import os
import shutil

import xarray as xr

from .mpi import MPITopology, _has_mpi

logger = logging.getLogger("ufs2arco")

class BatchLoader():
    """

    Note:
        * For now only works with GEFSDataset
        * This ends up being the same as the Naive version, except for the caching behavior,
          since it will for loop over all dates, but it clears the cache after batch_size dates
    """
    counter = 0
    data_counter = 0

    stop_event = None
    executor = None
    futures = None

    def __init__(
        self,
        dataset,
        batch_size,
        sample_dims,
        num_workers=0,
        max_queue_size=1,
        start=0,
        cache_dir=".",
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.counter = start
        self.data_counter = start
        self.outer_cache_dir = cache_dir

        # construct the sample indices
        # e.g. {"t0": [date1, date2], "fhrs": [0, 6], "member": [0, 1, 2]}
        self.sample_dims = sample_dims
        all_sample_iterations = {
            key: getattr(dataset, key)
            for key in sample_dims
        }
        self.sample_indices = list()
        for combo in itertools.product(*all_sample_iterations.values()):
            self.sample_indices.append(dict(zip(sample_dims, combo)))

        self.num_workers = num_workers
        assert max_queue_size > 0
        max_queue_size = min(max_queue_size, len(self))
        self.max_queue_size = max_queue_size

        # create a separate lock for each of the attributes
        # that get changed, so threads don't bump into each other
        # It's important to have separate locks so we can lock
        # the state of each attribute separately
        self.counter_lock = threading.Lock()
        self.data_counter_lock = threading.Lock()
        self.executor_lock = threading.Lock()
        if self.num_workers > 0:
            self.data_queue = queue.Queue(maxsize=max_queue_size)
            self.stop_event = threading.Event()

        self.restart(idx=start)

    @property
    def name(self):
        return str(type(self).__name__)

    def get_cache_dir(self, batch_idx):
        return f"{self.outer_cache_dir}/{self.name.lower()}-cache/{batch_idx}"

    def __len__(self) -> int:
        n_samples = len(self.sample_indices)
        n_batches = ceil(n_samples / self.batch_size)
        return n_batches

    def __iter__(self):
        with self.counter_lock:
            self.counter = 0

        # Always restart in the serial case
        if self.num_workers == 0:
            self.restart()
        else:
            # in the parallel case, we don't want to unnecessarily clear the queue,
            # so we only restart if we've been grabbing data willy nilly
            # and we've exceeded the queue size
            # Also we restart if the BatchLoader was previously shutdown and needs a kick start
            if self.stop_event.is_set() or self.data_counter > self.max_queue_size:
                self.restart()
        return self

    def __next__(self):
        """Note that self.counter is the counter for looping with e.g. enumerate
        (i.e., how much has been removed from the queue)
        whereas self.data_counter is keeping track of how many data items have been put in the queue
        """
        if self.counter < len(self):
            data = self.get_data()
            with self.counter_lock:
                self.counter += 1
            return data
        else:
            logger.debug(f"BatchLoader.__next__: counter > len(self)")
            raise StopIteration

    def _next_data(self):
        logger.debug(f"BatchLoader._next_data[{self.data_counter}]")

        if self.data_counter < len(self):
            st = self.data_counter * self.batch_size
            ed = st + self.batch_size
            batch_indices = self.sample_indices[st:ed]
            dlist = []
            cache_dir = self.get_cache_dir(self.data_counter)
            for these_dims in batch_indices:
                fds = self.dataset.open_single_dataset(cache_dir=cache_dir, **these_dims)
                dlist.append(fds)
            xds = xr.merge(dlist)
            return xds

        else:
            logger.debug(f"BatchLoader._next_data: data_counter > len(self)")
            raise StopIteration

    def generate(self):
        while not self.stop_event.is_set():
            try:
                data = self._next_data()
                self.data_queue.put(data)
                with self.data_counter_lock:
                    self.data_counter += 1
                logger.debug(f"done putting")
            except StopIteration:
                self.shutdown()

    def get_data(self):
        """Pull a batch of data from the queue"""
        logger.debug(f"BatchLoader.get_data")
        if self.num_workers > 0:
            data = self.data_queue.get()
            self.task_done()
            return data
        else:
            data = self._next_data()
            with self.data_counter_lock:
                self.data_counter += 1
            return data

    def clear_cache(self, batch_idx):
        cache_dir = self.get_cache_dir(batch_idx)
        if os.path.isdir(cache_dir):
            logger.debug(f"{self.name}: clearing {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)


    def task_done(self):
        self.data_queue.task_done()
        logger.debug(f"BatchLoader: marked task_done")

    def restart(self, idx=0, cancel=False, **kwargs):
        """Restart the :attr:`data_counter` and ThreadPoolExecutor to get ready for the pass through the data

        Args:
            cancel (bool): if True, cancel any remaining queue items/tasks with :meth:`.cancel`
        """
        logger.debug(f"BatchLoader.restart")

        # start filling the queue
        if self.num_workers > 0:

            if self.executor is not None:
                self.shutdown(cancel=cancel, **kwargs)
                self.stop_event.clear()

            with self.data_counter_lock:
                self.data_counter = idx

            with self.executor_lock:
                self.executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.num_workers,
                )
                self.futures = [
                    self.executor.submit(self.generate) for _ in range(self.num_workers)
                ]
        else:
            self.data_counter = idx


    def cancel(self):
        """Cancel any remaining workers/queue items by calling :meth:`get_data` until they
        can recognize that the stop_event has been set
        """
        # cancel the existing workers/queue to force a startover
        i = 1
        if self.num_workers > 0:
            while not self.data_queue.empty():
                logger.debug(f"BatchLoader.cancel: Queue not empty. (count, data_count) = ({self.counter}, {self.data_counter})... getting data {i}")
                self.get_data()
                i+=1

    def shutdown(self, cancel=False, **kwargs):
        """Shutdown the ThreadPoolExecutor.

        Args:
            cancel (bool): If true, cancel any remaining tasks...
                Don't do this right after a for loop though, since the for loop may not finish due to a deadlock
        """
        logger.debug(f"BatchLoader.shutdown")
        if self.num_workers > 0:
            self.stop_event.set()
            if cancel:
                self.cancel()
            with self.executor_lock:
                self.executor.shutdown(**kwargs)
                # set executor to None, so that if shutdown is called from within a loop
                # and the BatchLoader is restarted immediately after, we don't get a double shutdown call
                self.executor = None

    def find_my_region(self, xds):
        """Given a dataset, that's assumed to be a subset of the initial dataset,
        find the logical index values where this should be stored in the final zarr store

        Args:
            xds (xr.Dataset): with a subset of the data (i.e., a couple of initial conditions)

        Returns:
            region (dict): indicating the zarr region to store in, based on the initial condition indices
        """
        region = {k: slice(None, None) for k in xds.dims}
        for key in self.sample_dims:
            full_array = getattr(self.dataset, key) # e.g. all of the initial conditions
            batch_indices = [list(full_array).index(value) for value in xds[key].values]
            region[key] = slice(batch_indices[0], batch_indices[-1]+1)
        return region



class MPIBatchLoader(BatchLoader):
    """Make sure mpi4py and mpi4jax is installed
    """
    def __init__(
        self,
        dataset,
        sample_dims,
        mpi_topo,
        num_workers=0,
        max_queue_size=1,
        start=0,
        cache_dir=".",
    ):
        assert _has_mpi, f"{self.name}.__init__: Unable to import mpi4py, cannot use this class"

        self.topo = mpi_topo
        batch_size = self.topo.size
        self.data_per_process = 1
        self.local_batch_index = self.topo.rank*self.data_per_process
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            sample_dims=sample_dims,
            num_workers=num_workers,
            max_queue_size=max_queue_size,
            start=start,
            cache_dir=cache_dir,
        )
        logger.info(str(self))

    def __str__(self):
        myname = f"{__name__}.{self.name}"
        underline = "".join(["-" for _ in range(len(myname))])
        msg = f"\n{myname}\n{underline}\n"

        for key in ["local_batch_index", "data_per_process", "batch_size"]:
            msg += f"{key:<18s}: {getattr(self, key):02d}\n"

        msg += f"{'Total Samples':<18s}: {len(self.sample_indices)}\n"
        msg += f"{'Total Batches':<18s}: {len(self)}\n"
        msg += f"{'sample_dims':<18s}: {self.sample_dims}\n"
        return msg

    def get_cache_dir(self, batch_idx):
        return f"{self.outer_cache_dir}/{self.name.lower()}-cache/{self.topo.rank}/{batch_idx}"

    def _next_data(self):
        if self.data_counter < len(self):
            st = (self.data_counter * self.batch_size) + self.local_batch_index
            ed = st + self.data_per_process
            batch_indices = self.sample_indices[st:ed]
            cache_dir = self.get_cache_dir(self.data_counter)
            if len(batch_indices) > 0:
                dlist = []

                for these_dims in batch_indices:
                    fds = self.dataset.open_single_dataset(cache_dir=cache_dir, **these_dims)
                    dlist.append(fds)
                xds = xr.merge(dlist)
                return xds

            else:
                return None
        else:
            raise StopIteration
