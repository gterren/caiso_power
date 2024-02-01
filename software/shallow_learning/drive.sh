# No. Batches
for A in {0..3}; do
	for B in {0..3}; do
  		sbatch run.job $A $B;
  		sleep 5s;
	done;
done;

# No. Batches
for A in 0 1 2; do
	for B in 1; do
	  	for C in 0; do
  		  sbatch mpirun.job $A $B $C;
  		  sleep 5s;
  		done;
	done;
done;

# No. Batches
for A in {0..3}; do
	sbatch run.job $A;
  sleep 5s;
done;

# No. Batches
for A in 0 1; do
	sbatch run.job $A;
  sleep 5s;
done;
