#!/bin/bash

mkdir -p ./logs

python -c "from ufs2arco.driver import Driver; Driver('gefs.serial.yaml').run(overwrite=True)" > logs/serial.log 2>&1

echo " *** Serial GEFSDataset ran successfully ***"

mpiexec -n 2 python -c "from ufs2arco.driver import Driver; Driver('gefs.mpi.yaml').run(overwrite=True)"

echo " *** MPI GEFSDataset ran successfully ***"

mpiexec -n 2 python -c "from ufs2arco.driver import Driver; Driver('gefs.anemoi.yaml').run(overwrite=True)"

echo " *** MPI GEFSDataset -> Anemoi ran successfully ***"
