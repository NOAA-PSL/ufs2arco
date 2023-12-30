"""A class to pick up the pieces from a failed ReplayMover job

1. figure out the gaps in the dataset by finding the cycle dates that did not finish
2. create a list with all of these cycle dates
3. method to convert these cycle dates to time
4. run through the cycle list, one at a time.

"""

from os.path import isdir
from shutil import rmtree
import logging
import numpy as np
import xarray as xr
import pandas as pd

from ufs2arco import FV3Dataset, Timer

from replay_mover import ReplayMover1Degree, ReplayMoverQuarterDegree


class ReplayFiller:
    """Fill in missing data due to failed jobs from ReplayMover

    Args:
        mover (ReplayMover): the ReplayMover object used. Note that it uses this object to determine how many jobs (nodes) to use and where to store cache, etc.
    """
    # from slurm and the data
    cycles_to_fill = np.concatenate([
        pd.date_range("1995-09-26T00", "1995-12-26T18", freq="6h"),
        pd.date_range("1997-01-07T00", "1997-12-21T00", freq="6h"),
        pd.date_range("1999-09-23T06", "1999-12-16T06", freq="6h"),
        pd.date_range("2000-12-17T12", "2001-12-10T06", freq="6h"),
        pd.date_range("2003-06-29T12", "2003-12-05T12", freq="6h"),
        pd.date_range("2005-07-27T18", "2005-11-29T18", freq="6h"),
        pd.date_range("2007-05-11T00", "2007-11-24T18", freq="6h"),
        pd.date_range("2009-09-12T00", "2009-11-19T00", freq="6h"),
        pd.date_range("2011-10-30T06", "2011-11-14T06", freq="6h"),
        pd.date_range("2015-03-14T12", "2015-11-03T12", freq="6h"),
        pd.date_range("2021-06-21T00", "2021-10-18T00", freq="6h"),
        pd.date_range("2022-10-12T06", "2023-10-13T06", freq="6h"),
    ])

    # this was a necessary quick 2nd fill
    #cycles_to_fill = np.array([
    #    np.datetime64("1997-11-23T00"),
    #    np.datetime64("2015-03-15T06"),
    #    np.datetime64("2015-04-01T06"),
    #])


    @property
    def xcycles_to_fill(self):
        """cycles_to_fill in xarray.DataArray form for time accessors"""
        return xr.DataArray(self.cycles_to_fill, coords={"cycles": self.cycles_to_fill}, dims="cycles")

    @property
    def splits(self):
        """Figure out the subset of cycles each job will be working on"""
        return [int(x) for x in np.linspace(0, len(self.xcycles_to_fill), self.n_jobs+1)]

    @property
    def n_jobs(self):
        return min(self.mover.n_jobs, len(self.cycles_to_fill))

    def __init__(self, mover):
        assert isinstance(mover, (ReplayMover1Degree, ReplayMoverQuarterDegree))
        self.mover = mover


    def my_cycles_to_fill(self, job_id):
        """The cycle timestamps for this job

        Args:
            job_id (int): the slurm job id, determines cache storage location

        Returns:
            cycles_datetime (List[datetime]): with the cycle numbers to be processed by this slurm job
        """
        slices = [slice(st, ed) for st, ed in zip(self.splits[:-1], self.splits[1:])]
        xda = self.xcycles_to_fill.isel(cycles=slices[job_id])
        cycles_datetime = self.mover.npdate2datetime(xda)
        return cycles_datetime


    def run(self, job_id):

        walltime = Timer()
        localtime = Timer()

        walltime.start(f"Starting Job {job_id}")
        replay = FV3Dataset(path_in=self.mover.cached_path, config_filename=self.mover.config_filename)

        for cycle in self.my_cycles_to_fill(job_id):

            localtime.start(f"Reading {str(cycle)}")
            try:
                xds = replay.open_dataset(cycle, **self.mover.ods_kwargs(job_id))

                index = list(self.mover.xtime.values).index(xds.time.values[0])
                tslice = slice(index, index+2)
                print("index = ", index)

                replay.store_dataset(
                    xds,
                    region={
                        "time": tslice,
                        "pfull": slice(None, None),
                        "grid_yt": slice(None, None),
                        "grid_xt": slice(None, None),
                    },
                    storage_options=self.mover.storage_options,
                )

                # This is a hacky way to clear the cache, since we don't create a filesystem object
                del xds
                if isdir(self.mover.cache_storage(job_id)):
                    rmtree(self.mover.cache_storage(job_id), ignore_errors=True)

            except Exception as e:
                logging.exception(e)
                print(f"ReplayFiller.run({job_id}): Failed to store {str(cycle)}")
                pass
            localtime.stop()

        walltime.stop("Total Walltime")
