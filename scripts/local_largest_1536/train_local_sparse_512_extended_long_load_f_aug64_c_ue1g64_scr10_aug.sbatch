#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250GB
#SBATCH --job-name=train_local
#SBATCH --output=train_local.out

eval "$(conda shell.bash hook)"
conda activate UAGL

python3 ./local_pipeline/train_4cor.py --dataset_name satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train --val_positive_dist_threshold 512 --database_size 1536 --num_steps 300000 --two_stages --corr_level 4 --restore_ckpt logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_18-44-07-97a33213-80a2-4f50-9d85-9ad04d7df728/RHWF.pth --finetune --detach --augment_two_stages 64 --first_stage_ue --batch_size 4 --ue_aug_method shift --ue_shift 64 --ue_num_crops 10 --ue_shift_crops_types grid --exclude_val_region --augment img --perspective_max 16 --rotate_max 0.523599 --resize_max 0.3