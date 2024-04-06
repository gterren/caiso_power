# No. Batches
for A in 0 1 2; do
	for B in 0 1 2 3; do
	  for C in 1; do
  		sbatch run.job $A $B $C;
  		sleep 5s;
  	done;
	done;
done;

# No. Batches
for A in 0; do
	for B in 2; do
	  for C in 3; do
	    for D in {0..99}; do
        sbatch run.job $A $B $C $D;
        sleep 5s;
      done;
  	done;
	done;
done;

# No. Batches
for A in {0..3}; do
	sbatch run_largemem.job $A;
  sleep 5s;
done;

# No. Batches
for A in 0 1 2 3; do
	sbatch run.job $A;
  sleep 5s;
done;

# No. Batches
for A in 0 1; do
	for B in 0 1 2 3; do
  		sbatch run.job $A $B;
  		sleep 5s;
	done;
done;

