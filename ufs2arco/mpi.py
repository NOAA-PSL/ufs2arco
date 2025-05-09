import os
import logging
import warnings
from typing import Optional, Any, List

try:
    from mpi4py import MPI
    _has_mpi = True
except ImportError:
    _has_mpi = False
    warnings.warn(f"ufs2arco.mpi: Unable to import mpi4py, cannot use this module")

from .log import SimpleFormatter

logger = logging.getLogger("ufs2arco")

class MPITopology:
    """
    Handles MPI-based parallel processing topology.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_level: int = logging.INFO,
    ) -> None:
        """
        Initializes the MPI topology.

        Args:
            log_dir (Optional[str]): Directory for storing logs.
            log_level (int): Logging level.
        """

        assert _has_mpi, f"MPITopology.__init__: Unable to import mpi4py, cannot use this class"

        self.required_level = MPI.THREAD_MULTIPLE
        self.provided_level = MPI.Query_thread()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.root = 0
        self.pid = os.getpid()
        self.friends = tuple(ii for ii in range(self.size) if ii!=self.root)

        self._init_log(log_dir=log_dir, level=log_level)
        logger.info(str(self))


    def __str__(self):
        msg = f"\nMPITopology Summary\n" +\
            f"-------------------\n" +\
            f"comm: {self.comm.Get_name()}\n"
        for key in ["rank", "size"]:
            msg += f"{key:<18s}: {getattr(self, key):02d}\n"
        msg += f"{'pid':<18s}: {self.pid}\n"
        msg += f"{'log_dir':<18s}: {self.log_dir}\n"
        msg += f"{'logfile':<18s}: {self.logfile}\n"
        msg += "Thread Support\n"+\
            f"{'required_level':<18s}: {self.required_level}\n" +\
            f"{'provided_level':<18s}: {self.provided_level}\n"
        return msg


    def _init_log(self, log_dir, level=logging.INFO):
        self.log_dir = "./" if log_dir is None else log_dir
        if self.is_root:
            if not os.path.isdir(self.log_dir):
                os.makedirs(self.log_dir)
        self.comm.Barrier()
        self.logfile = f"{self.log_dir}/log.{self.rank:04d}.{self.size:04d}.out"

        logger.setLevel(level=level)
        formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)-7s] %(message)s")
        handler = logging.FileHandler(self.logfile, mode="w+")
        handler.setLevel(level=level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


    @property
    def is_root(self) -> bool:
        """
        Checks if the current process is the root process.

        Returns:
            bool: True if the current process is the root, False otherwise.
        """
        return self.rank == self.root


    def barrier(self) -> None:
        """
        Synchronizes all processes at a barrier.
        """
        self.comm.Barrier()

    def Reduce(self, local_array, result_buffer, op):
        self.comm.Reduce(local_array, result_buffer, op=op, root=self.root)

    def sum(self, local_array, result_buffer):
        self.Reduce(local_array, result_buffer, op=MPI.SUM)

    def max(self, local_array, result_buffer):
        self.Reduce(local_array, result_buffer, op=MPI.MAX)

    def min(self, local_array, result_buffer):
        self.Reduce(local_array, result_buffer, op=MPI.MIN)

    def any(self, local_array, result_buffer):
        self.Reduce(local_array, result_buffer, op=MPI.LOR)


class SerialTopology:
    """
    Looks and acts like the MPITopology, but works in cases when mpi is not importable.

    So... basically it's a logger
    """
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_level: int = logging.INFO,
    ) -> None:
        """
        Args:
            log_dir (Optional[str]): Directory for storing logs.
            log_level (int): Logging level.
        """

        self.root = 0
        self.pid = os.getpid()
        self.size = 1
        self.rank = 0

        self._init_log(log_dir=log_dir, level=log_level)
        logger.info(str(self))

    def __str__(self):
        msg = f"\nSerialTopology Summary\n" +\
            f"---------------------\n"
        msg += f"{'pid':<18s}: {self.pid}\n"
        msg += f"{'log_dir':<18s}: {self.log_dir}\n"
        msg += f"{'logfile':<18s}: {self.logfile}\n"
        return msg


    def _init_log(self, log_dir, level=logging.INFO):
        self.log_dir = "./" if log_dir is None else log_dir
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.logfile = f"{self.log_dir}/log.serial.out"

        logger.setLevel(level=level)
        formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)-7s] %(message)s")
        handler = logging.FileHandler(self.logfile, mode="w+")
        handler.setLevel(level=level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    @property
    def is_root(self) -> bool:
        return True

    def barrier(self) -> None:
        pass

    # since the communication doesn't need to happen
    # we can assume these results already exist
    # this is just for code compatibility without mpi
    def sum(self, local_array, result_buffer):
        result_buffer[:] = local_array

    def max(self, local_array, result_buffer):
        result_buffer[:] = local_array

    def min(self, local_array, result_buffer):
        result_buffer[:] = local_array

    def any(self, local_array, result_buffer):
        result_buffer[:] = local_array
