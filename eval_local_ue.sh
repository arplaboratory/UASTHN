#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate UAGL

submit_job_ensemble_aug() {
    local STDM=$1
    local IT0=$2
    local IT1=$3
    local MODEL=$4

    local SHIFT_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_e_aug.sbatch"

    sbatch --export=ALL,DC=512,USEED=0,STDM=$STDM,AGG=zero,CS=-1,IT0=$IT0,IT1=$IT1,MODEL=$MODEL $SHIFT_SCRIPT
}

submit_job_ensemble() {
    local STDM=$1
    local IT0=$2
    local IT1=$3
    local MODEL=$4

    local SHIFT_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_e.sbatch"

    sbatch --export=ALL,DC=512,USEED=0,STDM=$STDM,AGG=zero,CS=-1,IT0=$IT0,IT1=$IT1,MODEL=$MODEL $SHIFT_SCRIPT
}

submit_job_shift() {
    local ST=$1
    local SHIFT=$2
    local STDM=$3
    local IT0=$4
    local IT1=$5
    local MODEL=$6

    local SHIFT_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_s.sbatch"

    sbatch --export=ALL,DC=512,ST=$ST,SHIFT=$SHIFT,USEED=0,STDM=$STDM,AGG=zero,CS=-1,IT0=$IT0,IT1=$IT1,MODEL=$MODEL $SHIFT_SCRIPT
}

submit_job_shift_aug() {
    local ST=$1
    local SHIFT=$2
    local STDM=$3
    local IT0=$4
    local IT1=$5
    local DEG=$6
    local MODEL=$7

    if [ "$DEG" = 1 ]; then
        local SHIFT_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_s_aug.sbatch"
    elif [ "$DEG" = 2 ]; then 
        local SHIFT_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_val_pm.sbatch"
    else
        local SHIFT_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_val_pl.sbatch"
    fi

    sbatch --export=ALL,DC=512,ST=$ST,SHIFT=$SHIFT,USEED=0,STDM=$STDM,AGG=zero,CS=-1,IT0=$IT0,IT1=$IT1,MODEL=$MODEL $SHIFT_SCRIPT
}

submit_job_shift_mock() {
    local ST=$1
    local SHIFT=$2
    local STDM=$3
    local IT0=$4
    local IT1=$5
    local MODEL=$6

    local SHIFT_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_val_mock.sbatch"

    sbatch --export=ALL,DC=512,ST=$ST,SHIFT=$SHIFT,USEED=0,STDM=$STDM,AGG=zero,CS=-1,IT0=$IT0,IT1=$IT1,MODEL=$MODEL $SHIFT_SCRIPT
}

submit_job_baseline() {
    local IT0=$1
    local IT1=$2
    local MODEL=$3

    local BASELINE_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_val.sbatch"

    sbatch --export=ALL,DC=512,IT0=$IT0,IT1=$IT1,MODEL=$MODEL $BASELINE_SCRIPT
}

# ################################################################################################################################################################################Train Methods

# # w/o MCT
# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-04-27_00-45-46-07ebf9cb-0452-4163-a883-ef4a57b7a5a8"
# ST=grid
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# # w/ MCT - scratch - 5
# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-04-26_22-38-42-c8df956c-7e9b-4419-a1a7-38f119adb90a"
# ST=grid
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# # w/ MCT - scratch - 10
# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-02_21-52-43-21c6b5c1-c8e4-49cf-b9e7-40d028029219"
# ST=grid
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# # w/ MCT -finetune - 5
# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-05_01-06-03-864cd069-f433-47ca-b947-f98f9602a562"
# ST=grid
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# # w/ MCT -finetune - 10
# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-05_01-06-08-186a5cef-53b8-412a-9e9c-9321dbe27597"
# ST=grid
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# ################################################################################################################################################################################Train Methods

# # w/ MCT - scratch - 5
# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-10_05-44-08-893060f8-fc8c-4c07-a383-ed4554119eeb"
# ST=grid
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# # w/ MCT - scratch - 10
# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-10_06-12-05-0a34633f-9ae9-44a0-897b-bdf6e3396dad"
# ST=grid
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# # w/ MCT -finetune - 5
# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-10_01-06-50-04673ae2-b2e8-483f-8149-029839920a8a"
# ST=grid
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# # w/ MCT -finetune - 10
# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-10_01-15-15-6c285869-f86d-471f-9ffc-0fb5717fbc39"
# ST=grid
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# ############################################################################################################################################ ensemble

# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-04-27_00-45-46-07ebf9cb-0452-4163-a883-ef4a57b7a5a8"
# STDM=all
# IT0=6
# IT1=6
# submit_job_ensemble $STDM $IT0 $IT1 $MODEL

# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-10_10-30-02-0b459c62-e0b2-4944-b113-27487a3275aa"
# STDM=all
# IT0=6
# IT1=6
# submit_job_ensemble_aug $STDM $IT0 $IT1 $MODEL


# #####################################################################################################################TTA

# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-07_17-31-18-ac64936c-2d3d-4487-9a41-6d58fbdeff9c"
# ST=grid
# SHIFT=32
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-07_21-16-31-edca2aa6-6617-4aa2-9de7-5fc0d610cc02"
# ST=random
# SHIFT=32
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# #######################################64

# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-05_01-06-08-186a5cef-53b8-412a-9e9c-9321dbe27597"
# ST=grid
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-07_17-32-25-94fed2f0-d2f8-4d27-b643-1ce03ae6995b"
# ST=random
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# submit_job_shift $ST $SHIFT $STDM $IT0 $IT1 $MODEL

# ########################################################################################################################################aug

# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-07_17-31-08-f3d602e1-e43c-4180-b72d-df3e501df8f5"
# ST=grid
# SHIFT=32
# STDM=all
# IT0=6
# IT1=6
# DEG=1
# submit_job_shift_aug $ST $SHIFT $STDM $IT0 $IT1 $DEG $MODEL

# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-07_17-31-19-b353128c-c99d-40e5-aa5f-47bdb5125b70"
# ST=random
# SHIFT=32
# STDM=all
# IT0=6
# IT1=6
# DEG=1
# submit_job_shift_aug $ST $SHIFT $STDM $IT0 $IT1 $DEG $MODEL

# #######################################64

# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-07_17-31-49-99b36f45-3aae-4801-9375-d48605f0142a"
# ST=grid
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# DEG=1
# submit_job_shift_aug $ST $SHIFT $STDM $IT0 $IT1 $DEG $MODEL

# MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-07_17-32-25-f6509aad-2905-4be4-8083-3d79a7fed615"
# ST=random
# SHIFT=64
# STDM=all
# IT0=6
# IT1=6
# DEG=1
# submit_job_shift_aug $ST $SHIFT $STDM $IT0 $IT1 $DEG $MODEL
