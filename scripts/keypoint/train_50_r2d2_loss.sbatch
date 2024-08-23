#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250GB
#SBATCH --job-name=train_keypoint
#SBATCH --output=train_keypoint.out

eval "$(conda shell.bash hook)"
conda activate UASTHN

python3 ./keypoint_pipeline/myr2d2/train_key.py --dataset_name satellite_0_thermalmapping_135 --val_positive_dist_threshold 50 --database_size 512 --num_steps 200000 --disable_reliability --batch_size 8