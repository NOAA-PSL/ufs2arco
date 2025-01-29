import os
import logging
import warnings

try:
    from mpi4py import MPI
    _has_mpi = True

except:
    _has_mpi = False
    warnings.warn(f"graphufs.mpi: Unable to import mpi4py or mpi4jax, cannot use this module")

from .log import SimpleFormatter

class MPITopology():

    @property
    def is_root(self):
        return self.rank == self.root

    def __init__(self, log_dir=None, log_level=logging.INFO):

        assert _has_mpi, f"MPITopology.__init__: Unable to import mpi4py, cannot use this class"
        self.required_level = MPI.THREAD_MULTIPLE
        self.provided_level = MPI.Query_thread()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        #self.local_size = len(jax.local_devices())
        #self.node = self.rank // self.local_size
        #self.local_rank = (self.rank - self.node*self.local_size) % self.local_size
        self.root = 0
        self.pid = os.getpid()
        self.friends = tuple(ii for ii in range(self.size) if ii!=self.root)

        self._init_log(log_dir=log_dir, level=log_level)
        logging.info(str(self))


    def __str__(self):
        msg = "\nEnvironment Info\n" +\
            "----------------\n" +\
            jax.print_environment_info(return_string=True) +\
            "\n\n" +\
            f"MPITopology Summary\n" +\
            f"-------------------\n" +\
            f"comm: {self.comm.Get_name()}\n"
        for key in ["rank", "size"]:
            msg += f"{key:<18s}: {getattr(self, key):02d}\n"
        msg += f"{'pid':<18s}: {self.pid}\n"
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
        self.logfile = f"{self.log_dir}/log.{self.rank:02d}.{self.size:02d}.out"
        self.progress_file = f"{self.log_dir}/progress.{self.rank:02d}.{self.size:02d}.out"

        logging.basicConfig(
            level=level,
            filename=self.logfile,
            filemode="w+",
        )
        logger = logging.getLogger()
        formatter = SimpleFormatter(fmt="[%(relativeCreated)-7d s] [%(levelname)-7s] %(message)s")
        for handler in logger.handlers:
            handler.setFormatter(formatter)

        with open(self.progress_file, "w"):
            pass

    def bcast(self, x):
        return self.comm.bcast(x, root=self.root)
