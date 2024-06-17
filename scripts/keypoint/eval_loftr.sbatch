#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250GB
#SBATCH --job-name=eval_keypoint
#SBATCH --output=eval_keypoint.out

eval "$(conda shell.bash hook)"
conda activate UAGL

python3 ./keypoint_pipeline/myloftr/myevaluate.py --dataset_name satellite_0_thermalmapping_135 --eval_model logs/local_key/test --val_positive_dist_threshold 50 --database_size 512 --batch_size 1 --test