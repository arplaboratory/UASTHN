#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=16               # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --job-name=train_tgm
#SBATCH --output=train_tgm.out

eval "$(conda shell.bash hook)"
conda activate UASTHN

python3 global_pipeline/train_pix2pix.py --dataset_name=satellite_0_thermalmapping_135 --datasets_folder=datasets --train_batch_size 32 --lr 0.0002 --patience 40 --epochs_num 60 --G_net unet --G_loss_lambda 100.0 --D_net patchGAN --GAN_save_freq 5 --GAN_resize 768 768 --GAN_epochs_decay 20 --G_contrast manual --num_workers 16
