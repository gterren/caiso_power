#!/bin/bash -l
#SBATCH --job-name=R1L6M3-SLX-DL10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=13
#SBATCH --time=8:00:00
##SBATCH --mem=187G
##SBATCH --output=res.txt
##SBATCH -p default
##SBATCH --mail-type=BEGIN,END
##SBATCH --mail-user=guillermoterren@ucsb.edu

export PATH=$PATH:/home/gterren/anaconda3/bin

source activate oasis

cd /home/gterren/caiso_power/software/shallow_learning

#mpirun -np 10 python mpi_shallow_learning.py 3 $1 0 $2
#time mpirun -np 3 python mpi_shallow_learning.py $1 $2 $3
#time mpirun -np 3 python mpi_shallow_learning_v01.py 1 1 1 0 0 0 $1 $2 1 2
#mpirun -np 3 python val_shallow_learning_v02.py 1 1 1 0 0 0 1 0 $1 0
#python mpirun -np 3 val_shallow_learning_v02.py 1 1 1 0 0 0 1 0 $1 1
#python mpirun -np 3 val_shallow_learning_v02.py 1 1 1 0 0 0 1 1 $1 2

#python test_multitask_shallow_learning_debugging.py 1 1 1 0 0 0 1 1 1 0 0
#python test_multitask_shallow_learning_debugging.py 1 1 1 1 0 0 1 1 1 3 5
#python test_multitask_shallow_learning_debugging.py 1 1 1 1 0 0 1 1 1 2 5
python -W ignore test_multitask_shallow_learning_debugging.py 1 1 1 1 0 0 1 1 1 3 5
