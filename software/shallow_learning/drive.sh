# No. Batches
for A in 0 1 2; do
	for B in 0 1 2 3; do
  		sbatch mpirun.slurm $A $B;
  		sleep 5s;
	done;
done;

# No. Batches
for A in 0 1; do
	for B in 0 1; do
  		sbatch mpirun.slurm $A $B;
  		sleep 5s;
	done;
done;


# No. Batches
for A in {0..29}; do
	sbatch mpirun.slurm $A;
  	sleep 5s;
done;

