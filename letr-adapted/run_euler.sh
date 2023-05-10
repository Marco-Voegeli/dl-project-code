# The goal of this script is to run the code on the Euler cluster
# Before running this script, you should have the virtual environment with all the dependencies installed
# First, it will pull the latest version of the code from github
# Then, it removes the old lsf files and activate the virtual environment
# Finally, it will run the code on the cluster

# IMPORTANT : all the configurations are done in the config.json file

git pull
TAG=$(git rev-parse --short HEAD)
source ~/.bashrc
conda activate deepl
cp exp/res101_stage2_516_from_a1_with_170_epochs/checkpoints/checkpoint.pth exp/res101_stage2/checkpoints/checkpoint.pth

sbatch < run.sh
