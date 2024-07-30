# No. Batches
for A in 0 1 2; do
	for B in 1; do
	  for C in {0..34}; do
  		sbatch run.job $A $B $C;
  		sleep 5s;
  	done;
	done;
done;

# No. Batches
for A in 2; do
	for B in 1; do
	  for C in {0..34}; do
	    for D in {0..5}; do
        sbatch run.job $A $B $C $D;
        sleep 4s;
      done;
  	done;
	done;
done;

for A in {0..2}; do
	for B in {0..59}; do
  	sbatch run.job $A $B;
  	sleep 5s;
  done;
done;

# No. Batches
for A in 0 1 2 3; do
	sbatch run.job $A;
  sleep 5s;
done;

# No. Batches
for A in 1 2; do
	for B in {0..79}; do
  		sbatch run.job $A $B;
  		sleep 5s;
	done;
done;

