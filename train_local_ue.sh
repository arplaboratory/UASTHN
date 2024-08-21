#!/bin/bash

# Ensemble
# Baseline without UE directly uses the model with seed=0
# Step 1: Train one-stage models with different seeds
for i in {0..4}
do
    sbatch --export=ALL,SEED="$i",DC=512 scripts/local_largest_1536/train_local_sparse_extended_long_ihn.sbatch
    sbatch --export=ALL,SEED="$i",DC=256 scripts/local_largest_1536/train_local_sparse_extended_long_ihn.sbatch
    sbatch --export=ALL,SEED="$i",DC=128 scripts/local_largest_1536/train_local_sparse_extended_long_ihn.sbatch
    echo "$i"
done

# Step 2: Train two-stage models with different seeds
for i in {0..4}
do
    # Change MODEL_FOLDER to the folder of the model with seed=0
    sbatch --export=ALL,SEED="$i",DC=512,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_sthn.sbatch
    sbatch --export=ALL,SEED="$i",DC=256,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_sthn.sbatch
    sbatch --export=ALL,SEED="$i",DC=128,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_sthn.sbatch
    echo "$i"
done

# Step 3: Train uncertainty estimation models
# TTA
# One-stage models
# Change MODEL_FOLDER to the folder of the base model
sbatch --export=ALL,DC=512,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_ihn_c_ue1r32.sbatch
sbatch --export=ALL,DC=256,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_ihn_c_ue1r32.sbatch
sbatch --export=ALL,DC=128,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_ihn_c_ue1r32.sbatch

# Two-stage models
# Change MODEL_FOLDER to the folder of the base model
sbatch --export=ALL,DC=512,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_load_f_aug64_c_ue1r32.sbatch
sbatch --export=ALL,DC=256,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_load_f_aug64_c_ue1r32.sbatch
sbatch --export=ALL,DC=128,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_load_f_aug64_c_ue1r32.sbatch

# DM
# One-stage models
# Change MODEL_FOLDER to the folder of the base model
sbatch --export=ALL,DC=512 scripts/local_largest_1536/train_local_sparse_extended_long_ihn_d.sbatch
sbatch --export=ALL,DC=256 scripts/local_largest_1536/train_local_sparse_extended_long_ihn_d.sbatch
sbatch --export=ALL,DC=128 scripts/local_largest_1536/train_local_sparse_extended_long_ihn_d.sbatch

# Two-stage models
# Change MODEL_FOLDER to the folder of the base model
sbatch --export=ALL,DC=512,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_load_f_aug64_sthn_d.sbatch
sbatch --export=ALL,DC=256,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_load_f_aug64_sthn_d.sbatch
sbatch --export=ALL,DC=128,MODEL=$MODEL_FOLDER scripts/local_largest_1536/train_local_sparse_extended_long_load_f_aug64_sthn_d.sbatch

# DE
# Use the models trained in step 1 and 2. Put trained models in local_pipeline/ensembles