#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=250GB
#SBATCH --job-name=train_local
#SBATCH --output=train_local.out

eval "$(conda shell.bash hook)"
conda activate UAGL

python3 ./local_pipeline/train_4cor.py --dataset_name satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train --val_positive_dist_threshold 512 --database_size 1536 --num_steps 200000 --corr_level 4  --first_stage_ue --batch_size 8 --ue_aug_method shift --ue_shift 32 --ue_num_crops 5 --ue_shift_crops_types random --exclude_val_region --ue_agg mean --lr 5e-5 --restore_ckpt logs/local_he/satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-03-16_10-31-15-88758144-cad2-4613-ba15-d28694151775/STHN.pth --augment img --perspective_max 16 --rotate_max 0.523599 --resize_max 0.3