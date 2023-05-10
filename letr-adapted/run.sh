#!/bin/bash

#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=3300m

#SBATCH --gpus-per-node=3
#SBATCH --gres=gpumem:20g
#SBATCH --time=0-23:59:59


#SBATCH --job-name=a0
#SBATCH --output=a0.out


#SBATCH --mail-type=ALL                                 # notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-user=your-ethz-username@ethz.ch  # who to send email notification for job stats changes


echo "Running A0"

rm -rf exp/a0/
bash script/train/a0_train_stage1.sh  a0

echo "Done rendering"

exit 0
