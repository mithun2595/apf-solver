#!/bin/bash
make clean
make
mpirun -np 8 ./apf -n 400 -i 200 -x 1 -y 8
