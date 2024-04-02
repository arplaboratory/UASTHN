#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --job-name=train_local
#SBATCH --output=train_local.out

eval "$(conda shell.bash hook)"
conda activate UAGL

python3 ./local_pipeline/train_4cor.py --dataset_name satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train --val_positive_dist_threshold 256 --database_size 2560 --num_steps 300000 --two_stages --corr_level 4 --restore_ckpt logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-03-31_21-03-31-693753ac-1d06-4b47-829b-b7a40d887d72/UAGL.pth --finetune --detach --augment_two_stages 64