#!/bin/bash
#### this a bash script which we write out and then submit to the batch queue
# This script is intepreted by the Bourne Shell, sh
#
#SBATCH --account=csd562
#SBATCH --job-name=apf-strong-scale-48
#SBATCH --output="strong_scaling-96-1024.out"
#SBATCH --partition="compute"
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=16
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

ibrun -np 96 ./apf -n 1800 -i 2000 -x 2 -y 48 
ibrun -np 192 ./apf -n 1800 -i 2000 -x 2 -y 96 
ibrun -np 240 ./apf -n 1800 -i 2000 -x 2 -y 120 
ibrun -np 384 ./apf -n 1800 -i 2000 -x 2 -y 192 
ibrun -np 480 ./apf -n 1800 -i 2000 -x 2 -y 240 
ibrun -np 960 ./apf -n 1800 -i 2000 -x 2 -y 480 
ibrun -np 1024 ./apf -n 1800 -i 2000 -x 2 -y 512 


#ibrun -np 48 ./apf -n 1800 -i 2000 -x 4 -y 12
#ibrun -np 96 ./apf -n 1800 -i 2000 -x 4 -y 24 
#ibrun -np 192 ./apf -n 1800 -i 2000 -x 4 -y 48 
#ibrun -np 240 ./apf -n 1800 -i 2000 -x 4 -y 60
#ibrun -np 384 ./apf -n 1800 -i 2000 -x 2 -y 192 -k
#ibrun -np 480 ./apf -n 1800 -i 2000 -x 4 -y 120 
#ibrun -np 960 ./apf -n 1800 -i 2000 -x 4 -y 240 
#ibrun -np 1024 ./apf -n 1800 -i 2000 -x 4 -y 256 


#ibrun -np 96 ./apf -n 400 -i 500 -x 1 -y 96
#ibrun -np 96 ./apf -n 400 -i 500 -x 2 -y 48 
#ibrun -np 96 ./apf -n 400 -i 500 -x 3 -y 32
#ibrun -np 96 ./apf -n 400 -i 500 -x 4 -y 24
#ibrun -np 96 ./apf -n 400 -i 500 -x 6 -y 16
#ibrun -np 96 ./apf -n 400 -i 500 -x 8 -y 12
#ibrun -np 96 ./apf -n 400 -i 500 -y 1 -x 96
#ibrun -np 96 ./apf -n 400 -i 500 -y 2 -x 48 
#ibrun -np 96 ./apf -n 400 -i 500 -y 3 -x 32
#ibrun -np 96 ./apf -n 400 -i 500 -y 4 -x 24
#ibrun -np 96 ./apf -n 400 -i 500 -y 6 -x 16
#ibrun -np 96 ./apf -n 400 -i 500 -y 8 -x 12

