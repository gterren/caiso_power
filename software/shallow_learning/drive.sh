# No. Batches
for A in 0 1 2; do
	for B in 0 1 2 3; do
	  for C in 3; do
  		sbatch run.job $A $B $C;
  		sleep 5s;
  	done;
	done;
done;

# No. Batches
for A in 0; do
	for B in 3; do
	  for C in 3; do
	    for D in {0..104}; do
        sbatch run.job $A $B $C $D;
        sleep 5s;
      done;
  	done;
	done;
done;

# No. Batches
for A in {0..63}; do
	sbatch run_braid.job $A;
  sleep 5s;
done;

# No. Batches
for A in 0 1 2 3; do
	sbatch run.job $A;
  sleep 5s;
done;

# No. Batches
for A in 0 1 2; do
  for B in {0..63}; do
    sbatch run_braid.job $A $B;
  	sleep 5s;
	done;
done;

