#!/bin/bash
#### this a bash script which we write out and then submit to the batch queue
# This script is intepreted by the Bourne Shell, sh
#
#SBATCH --account=csd562
#SBATCH --job-name=apf-strong-scale-48
#SBATCH --output="apf-optimum-geometry-for-96.out"
#SBATCH --partition="compute"
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=6
#SBATCH --export=ALL
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu
#SBATCH -t 00:03:00
# Commands go here

#ibrun -np 24 ./apf -n 1800 -i 2000 -x 2 -y 12
#ibrun -np 36 ./apf -n 1800 -i 2000 -x 2 -y 18
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 2 -y 24
#ibrun -np 60 ./apf -n 1800 -i 2000 -x 2 -y 30
#ibrun -np 72 ./apf -n 1800 -i 2000 -x 2 -y 36
#ibrun -np 84 ./apf -n 1800 -i 2000 -x 2 -y 42
#ibrun -np 12 ./apf -n 1800 -i 2000 -x 2 -y 6 
#ibrun -np 24 ./apf -n 1800 -i 2000 -x 2 -y 12 
ibrun -np 96 ./apf -n 1800 -i 2000 -x 1 -y 96 
ibrun -np 96 ./apf -n 1800 -i 2000 -x 2 -y 48 
ibrun -np 96 ./apf -n 1800 -i 2000 -x 3 -y 32
ibrun -np 96 ./apf -n 1800 -i 2000 -x 4 -y 24
ibrun -np 96 ./apf -n 1800 -i 2000 -x 6 -y 16  
ibrun -np 96 ./apf -n 1800 -i 2000 -x 8 -y 12 
ibrun -np 96 ./apf -n 1800 -i 2000 -x 12 -y 8 
ibrun -np 96 ./apf -n 1800 -i 2000 -x 16 -y 6 
ibrun -np 96 ./apf -n 1800 -i 2000 -x 24 -y 4 
ibrun -np 96 ./apf -n 1800 -i 2000 -x 32 -y 3
ibrun -np 96 ./apf -n 1800 -i 2000 -x 48 -y 2
ibrun -np 96 ./apf -n 1800 -i 2000 -x 96 -y 1  








