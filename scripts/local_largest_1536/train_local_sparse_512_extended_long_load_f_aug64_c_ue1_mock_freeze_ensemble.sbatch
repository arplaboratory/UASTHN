#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250GB
#SBATCH --job-name=train_local
#SBATCH --output=train_local.out

eval "$(conda shell.bash hook)"
conda activate UAGL

python3 ./local_pipeline/train_4cor.py --dataset_name satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train --val_positive_dist_threshold 512 --database_size 1536 --num_steps 100000 --two_stages --corr_level 4 --restore_ckpt logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-04-27_00-45-46-07ebf9cb-0452-4163-a883-ef4a57b7a5a8/UAGL.pth --finetune --detach --augment_two_stages 64 --first_stage_ue --batch_size 4 --ue_num_crops 10 --exclude_val_region --ue_mock --ue_mock_freeze --ue_method ensemble