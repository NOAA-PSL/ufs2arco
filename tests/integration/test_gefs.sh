#!/bin/bash

mkdir -p ./logs

python -c "from ufs2arco.driver import Driver; Driver('gefs.serial.yaml').run(overwrite=True)" > logs/serial.log 2>&1

echo " *** Done running Serial GEFSDataset ***"
echo "     tail -n 1 logs/log.serial.out"
tail -n 1 logs/log.serial.out
echo ""

mpiexec -n 2 python -c "from ufs2arco.driver import Driver; Driver('gefs.mpi.yaml').run(overwrite=True)"

echo " *** Done running MPI GEFSDataset ***"
echo "     tail -n 1 logs/mpi/log.0000.0002.out"
tail -n 1 logs/mpi/log.0000.0002.out
echo ""


mpiexec -n 2 python -c "from ufs2arco.driver import Driver; Driver('gefs.anemoi.yaml').run(overwrite=True)"

echo " *** Done running MPI GEFSDataset -> Anemoi ***"
echo "     tail -n 1 logs/anemoi/log.0000.0002.out"
tail -n 1 logs/anemoi/log.0000.0002.out
