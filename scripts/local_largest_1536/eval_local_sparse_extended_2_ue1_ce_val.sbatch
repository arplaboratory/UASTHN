#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=5
#SBATCH --mem=250GB
#SBATCH --job-name=eval_local
#SBATCH --output=eval_local.out

eval "$(conda shell.bash hook)"
conda activate UAGL

CROP=$SLURM_ARRAY_TASK_ID

if [ "$CROP" -le 5 ]; then
    BAN=8
else
    BAN=4
fi

python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_train --eval_model logs/local_he/$MODEL --val_positive_dist_threshold $DC --lev0 --database_size 1536 --corr_level 4 --two_stages --fine_padding 32 --first_stage_ue --batch_size $BAN --ue_aug_method shift --ue_num_crops $CROP --ue_shift_crops_types $ST --ue_shift $SHIFT --ue_seed $USEED --ue_std_method $STDM --ue_agg $AGG --check_step $CS --ue_method augment_ensemble --ue_ensemble_load_models $EN --arch $ARCH --ue_combine $COM