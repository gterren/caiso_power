#!/bin/bash -l
#SBATCH --job-name=L6M3_1111-SL0-DL0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=48:00:00
##SBATCH --mem=187G
##SBATCH --output=res.txt
##SBATCH -p default
##SBATCH --mail-type=BEGIN,END
##SBATCH --mail-user=guillermoterren@ucsb.edu

export PATH=$PATH:/home/gterren/anaconda3/bin

source activate oasis

cd /home/gterren/caiso_power/software/shallow_learning

#time mpirun -np 3 python -W ignore val_shallow_learning_v02.py 0 0 1 1 $1 1
#time mpirun -np 3 python -W ignore val_shallow_learning_v02.py 0 0 1 1 2 0
time mpirun -np 3 python -W ignore val_shallow_learning_v03.py 0 0 1 1 0 0 0
