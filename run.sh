#!/bin/bash
make clean
make
mpirun -np 4 ./apf -n 400 -i 2000 -x 1 -y 4
