#!/bin/bash -l
#SBATCH --job-name=CV_EM4-FE2
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=10
#SBATCH --time=48:00:00
#SBATCH --mem=187G   
##SBATCH --output=res.txt        
##SBATCH -p batch
##SBATCH --mail-type=BEGIN,END
##SBATCH --mail-user=guillermoterren@ucsb.edu

export PATH=$PATH:/home/gterren/anaconda3/bin

cd  /home/gterren/anaconda3/bin
source activate oasis

cd /home/gterren/caiso_power/software

#python CAISO_API_for_renewable_generation_and_demand.py 0 2020 1 1
#python NOAA_API_for_weather.py 0 2020 1 1
#python CAISO_API_for_local_marginal_prices.py 0 2020 1 1

#python cv_expert_models.py 4 0 $1 $2
#python cv_expert_models.py 3 0 $1 0
python cv_expert_models.py 4 2 $1 $2
#2-3_0-0-1_0-0
#python test_single_expert_models.py 2 3 0 0 1 0 0
