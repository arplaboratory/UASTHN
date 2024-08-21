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
conda activate UASTHN

python3 ./local_pipeline/train_4cor.py --dataset_name satellite_0_thermalmapping_135_train --val_positive_dist_threshold $DC --database_size 1536 --num_steps 300000 --two_stages --corr_level 4 --restore_ckpt logs/local_he/$MODEL/UASTHN.pth --finetune --detach --augment_two_stages 64 --first_stage_ue --batch_size 16 --exclude_val_region --ue_method single