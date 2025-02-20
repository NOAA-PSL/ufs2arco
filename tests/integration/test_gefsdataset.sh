#!/bin/bash


python -c "from ufs2arco.driver import Driver; Driver('gefsdataset.serial.yaml').run(overwrite=True)" > gefsdataset.serial.log 2>&1

echo " *** Serial GEFSDataset ran successfully ***"

mpiexec -n 2 python -c "from ufs2arco.driver import Driver; Driver('gefsdataset.mpi.yaml').run(overwrite=True)"

echo " *** MPI GEFSDataset ran successfully ***"
