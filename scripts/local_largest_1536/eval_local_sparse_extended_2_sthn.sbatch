#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250GB
#SBATCH --job-name=eval_local
#SBATCH --output=eval_local.out

eval "$(conda shell.bash hook)"
conda activate UASTHN

python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_train --eval_model logs/local_he/$MODEL --val_positive_dist_threshold $DC --lev0 --database_size 1536 --corr_level 4 --two_stages --fine_padding 32 --test