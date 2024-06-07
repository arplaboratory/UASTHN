#!/bin/bash

# Ensemble
# Baseline without UE directly uses the model with seed=0
for i in {0..4}
do
    sbatch --export=ALL,SEED="$i" scripts/local_largest_1536/train_local_sparse_512_extended_long.sbatch
    sbatch --export=ALL,SEED="$i" scripts/local_largest_1536/train_local_sparse_512_extended_long_dhn.sbatch
    sbatch --export=ALL,SEED="$i" scripts/local_largest_1536/train_local_sparse_256_extended_long_load_f_aug64_c.sbatch
    echo "$i"
done

for i in {0..4}
do
    sbatch --export=ALL,SEED="$i" scripts/local_largest_1536/train_local_sparse_256_extended_long.sbatch
    sbatch --export=ALL,SEED="$i" scripts/local_largest_1536/train_local_sparse_256_extended_long_load_f_aug64_c.sbatch
    sbatch --export=ALL,SEED="$i" scripts/local_largest_1536/train_local_sparse_256_extended_long_dhn.sbatch
    echo "$i"
done

for i in {0..4}
do
    sbatch --export=ALL,SEED="$i" scripts/local_largest_1536/train_local_sparse_128_extended_long.sbatch
    sbatch --export=ALL,SEED="$i" scripts/local_largest_1536/train_local_sparse_128_extended_long_load_f_aug64_c.sbatch
    sbatch --export=ALL,SEED="$i" scripts/local_largest_1536/train_local_sparse_128_extended_long_dhn.sbatch
    echo "$i"
done

# TTA ablation study
sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_load_f_aug64_c_ue1g16_ft5.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_load_f_aug64_c_ue1g32_ft5.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_load_f_aug64_c_ue1g64_ft5.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_load_f_aug64_c_ue1r16_ft5.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_load_f_aug64_c_ue1r32_ft5.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_load_f_aug64_c_ue1r64_ft3.sbatch

sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_load_f_aug64_c_ue1r32_ft3.sbatch

# Direct Modeling
sbatch scripts/local_largest_1536/train_local_sparse_128_extended_long_dhn_single.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_128_extended_long_load_f_aug64_c_single.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_128_extended_long_single.sbatch

sbatch scripts/local_largest_1536/train_local_sparse_256_extended_long_dhn_single.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_256_extended_long_load_f_aug64_c_single.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_256_extended_long_single.sbatch

sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_dhn_single.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_single.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_load_f_aug64_c_single.sbatch

# TTA test
sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_dhn_ue1r32.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_256_extended_long_dhn_ue1r32.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_128_extended_long_dhn_ue1r32.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_512_extended_long_ue1r32.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_256_extended_long_ue1r32.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_128_extended_long_ue1r32.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_256_extended_long_load_f_aug64_c_ue1r32.sbatch
sbatch scripts/local_largest_1536/train_local_sparse_128_extended_long_load_f_aug64_c_ue1r32.sbatch