from math import ceil
import numpy as np

import logging
import threading
import concurrent
import queue

import os
import shutil

import xarray as xr

from .mpi import MPITopology, _has_mpi

class BatchLoader():
    """

    Note:
        * Assumes parallelism is over the initial condition dates
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
        num_workers=0,
        max_queue_size=1,
        start=0,
        cache_dir="./batchcache",
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.counter = start
        self.data_counter = start
        self.cache_dir = cache_dir

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
    def dates(self):
        """Returns dates of all initial conditions"""
        return self.dataset.dates

    @property
    def name(self):
        return str(type(self).__name__)

    def __len__(self) -> int:
        n_dates = len(self.dates)
        n_batches = ceil(n_dates / self.batch_size)
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
            logging.debug(f"BatchLoader.__next__: counter > len(self)")
            raise StopIteration

    def _next_data(self):
        logging.debug(f"BatchLoader._next_data[{self.data_counter}]")

        if self.data_counter < len(self):
            st = self.data_counter * self.batch_size
            ed = st + self.batch_size
            batch_dates = self.dates[st:ed]
            dlist = []
            for date in batch_dates:
                fds = self.dataset.open_single_initial_condition(date, cache_dir=self.cache_dir)
                dlist.append(fds)
            xds = xr.merge(dlist)

            # seems like the most threadsafe place to clear the cache directory
            if os.path.isdir(self.cache_dir):
                shutil.rmtree(self.cache_dir, ignore_errors=True)
            return xds

        else:
            logging.debug(f"BatchLoader._next_data: data_counter > len(self)")
            raise StopIteration

    def generate(self):
        while not self.stop_event.is_set():
            try:
                data = self._next_data()
                self.data_queue.put(data)
                with self.data_counter_lock:
                    self.data_counter += 1
                logging.debug(f"done putting")
            except StopIteration:
                self.shutdown()

    def get_data(self):
        """Pull a batch of data from the queue"""
        logging.debug(f"BatchLoader.get_data")
        if self.num_workers > 0:
            data = self.data_queue.get()
            self.task_done()
            return data
        else:
            data = self._next_data()
            with self.data_counter_lock:
                self.data_counter += 1
            return data

    def task_done(self):
        self.data_queue.task_done()
        logging.debug(f"BatchLoader: marked task_done")

    def restart(self, idx=0, cancel=False, **kwargs):
        """Restart the :attr:`data_counter` and ThreadPoolExecutor to get ready for the pass through the data

        Args:
            cancel (bool): if True, cancel any remaining queue items/tasks with :meth:`.cancel`
        """
        logging.debug(f"BatchLoader.restart")

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
                logging.debug(f"BatchLoader.cancel: Queue not empty. (count, data_count) = ({self.counter}, {self.data_counter})... getting data {i}")
                self.get_data()
                i+=1

    def shutdown(self, cancel=False, **kwargs):
        """Shutdown the ThreadPoolExecutor.

        Args:
            cancel (bool): If true, cancel any remaining tasks...
                Don't do this right after a for loop though, since the for loop may not finish due to a deadlock
        """
        logging.debug(f"BatchLoader.shutdown")
        if self.num_workers > 0:
            self.stop_event.set()
            if cancel:
                self.cancel()
            with self.executor_lock:
                self.executor.shutdown(**kwargs)
                # set executor to None, so that if shutdown is called from within a loop
                # and the BatchLoader is restarted immediately after, we don't get a double shutdown call
                self.executor = None


class MPIBatchLoader(BatchLoader):
    """Make sure mpi4py and mpi4jax is installed
    """
    def __init__(
        self,
        dataset,
        batch_size,
        mpi_topo,
        num_workers=0,
        max_queue_size=1,
        start=0,
        cache_dir="./batchcache",
    ):
        assert _has_mpi, f"{self.name}.__init__: Unable to import mpi4py, cannot use this class"

        self.topo = mpi_topo
        self.data_per_process = batch_size // self.topo.size
        self.local_batch_index = self.topo.rank*self.data_per_process
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            max_queue_size=max_queue_size,
            start=start,
            cache_dir=cache_dir,
        )
        self.cache_dir = os.path.join(cache_dir, f"{mpi_topo.rank}")
        logging.info(str(self))

        if self.data_per_process*self.topo.size != batch_size:
            logging.warning(f"{self.name}.__init__: batch_size = {batch_size} not divisible by MPI Size = {self.topo.size}")
            logging.warning(f"{self.name}.__init__: some data will be skipped in each batch")

        # need to see if this is still true
        if batch_size > 1 and not drop_last:
            logging.warning(f"{self.name}.__init__: with batch_size>1 and drop_last=False, some MPI processes may grab incorrect indices in last batch. Expect an error at the end of the dataset")

    def __str__(self):
        myname = f"{__name__}.{self.name}"
        underline = "".join(["-" for _ in range(len(myname))])
        msg = f"\n{myname}\n{underline}\n"

        for key in ["local_batch_index", "data_per_process", "batch_size"]:
            msg += f"{key:<18s}: {getattr(self, key):02d}\n"
        return msg


    def _next_data(self):
        if self.data_counter < len(self):
            st = (self.data_counter * self.batch_size) + self.local_batch_index
            ed = st + self.data_per_process
            batch_dates = self.dates[st:ed]
            if len(batch_dates) > 0:
                dlist = []
                for date in batch_dates:
                    fds = self.open_single_initial_condition(date, cache_dir=self.cache_dir)
                    dlist.append(fds)
                xds = xr.merge(dlist)

                # seems like the most threadsafe place to clear the cache directory
                if os.path.isdir(self.cache_dir):
                    shutil.rmtree(self.cache_dir, ignore_errors=True)
                return xds
            else:
                return None

        else:
            raise StopIteration
