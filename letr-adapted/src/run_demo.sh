#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=20G

#SBATCH --gpus-per-node=1
#SBATCH --time=0-3:59:59

#SBATCH --job-name=demo
#SBATCH --output=demo.out


echo "Running demo"


EXP_PATH=/cluster/scratch/atabin/LETR_euler/exp
echo "EXP_PATH: $EXP_PATH"
echo "Copying checkpoints to $EXP_PATH/checkpoints"
cp $EXP_PATH/res50_stage1_415_24h/checkpoints/checkpoint.pth $EXP_PATH/checkpoints/checkpoint_res50_s1.pth
cp $EXP_PATH/res50_stage2_from_a0_with_200_epochs/checkpoints/checkpoint.pth $EXP_PATH/checkpoints/checkpoint_res50_s2.pth
cp $EXP_PATH/a4/checkpoints/checkpoint.pth $EXP_PATH/checkpoints/checkpoint_res50_s3.pth

cp $EXP_PATH/res101_stage1_718_24h/checkpoints/checkpoint.pth $EXP_PATH/checkpoints/checkpoint_res101_s1.pth
cp $EXP_PATH/res101_stage2_516_from_a1_with_170_epochs/checkpoints/checkpoint.pth $EXP_PATH/checkpoints/checkpoint_res101_s2.pth
cp /cluster/scratch/atabin/LETR_euler_a5/exp/a5/checkpoints/checkpoint.pth $EXP_PATH/checkpoints/checkpoint_res101_s3.pth

python3 demo.py

echo "Done running demo"

exit 0
