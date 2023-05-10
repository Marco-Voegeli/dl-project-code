#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G

#SBATCH --gpus-per-node=1
#SBATCH --time=0-3:59:59

#SBATCH --job-name=demo_letr
#SBATCH --output=demo_letr.out


echo "Running LETR demo"

python3 demo_letr.py

echo "Done running demo"

exit 0
