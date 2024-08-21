#!/bin/bash

submit_job_ensemble() {
    local DC=$1
    local EN=$2
    local ARCH=$3
    local MODEL=$4
    local METHOD=$5

    if [ "$METHOD" = 1 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_e_sthn_val.sbatch"
    elif [ "$METHOD" = 2 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_e_sthn.sbatch"
    elif [ "$METHOD" = 3 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_e_ihn.sbatch"
    else
        exit 0
    fi

    sbatch --export=ALL,DC=$DC,USEED=0,STDM=all,AGG=mean,CS=-1,MODEL=$MODEL,EN=$EN,ARCH=$ARCH $EVAL_SCRIPT
}

submit_job_single() {
    local DC=$1
    local ARCH=$2
    local MODEL=$3
    local METHOD=$4

    if [ "$METHOD" = 1 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_d_sthn_val.sbatch"
    elif [ "$METHOD" = 2 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_d_sthn.sbatch"
    elif [ "$METHOD" = 3 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_d_ihn.sbatch"
    else
        exit 0
    fi

    sbatch --export=ALL,DC=$DC,USEED=0,STDM=all,AGG=mean,CS=-1,MODEL=$MODEL,ARCH=$ARCH $EVAL_SCRIPT
}

submit_job_crop() {
    local DC=$1
    local ST=$2
    local SHIFT=$3
    local ARCH=$4
    local MODEL=$5
    local METHOD=$6

    if [ "$METHOD" = 1 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_c_sthn_val.sbatch"
    elif [ "$METHOD" = 2 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_c_sthn.sbatch"
    elif [ "$METHOD" = 3 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_c_ihn.sbatch"
    else
        exit 0
    fi

    sbatch --export=ALL,DC=$DC,ST=$ST,SHIFT=$SHIFT,USEED=0,STDM=all,AGG=zero,CS=-1,ARCH=$ARCH,MODEL=$MODEL $EVAL_SCRIPT
}

submit_job_crop_ensemble() {
    local DC=$1
    local ST=$2
    local SHIFT=$3
    local ARCH=$4
    local EN=$5
    local COM=$6
    local MODEL=$7
    local METHOD=$8

    if [ "$METHOD" = 1 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_ce_sthn_val.sbatch"
    elif [ "$METHOD" = 2 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_ce_sthn.sbatch"
    elif [ "$METHOD" = 3 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ue1_ce_ihn.sbatch"
    else
        exit 0
    fi

    sbatch --export=ALL,DC=$DC,ST=$ST,SHIFT=$SHIFT,USEED=0,STDM=all,AGG=zero,CS=-1,ARCH=$ARCH,EN=$EN,COM=$COM,MODEL=$MODEL $EVAL_SCRIPT
}

submit_job_baseline() {
    local DC=$1
    local MODEL=$2
    local METHOD=$3

    if [ "$METHOD" = 1 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_sthn_val.sbatch"
    elif [ "$METHOD" = 2 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_sthn.sbatch"
    elif [ "$METHOD" = 3 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_extended_2_ihn.sbatch"
    else
        exit 0
    fi

    sbatch --export=ALL,DC=$DC,MODEL=$MODEL $BASELINE_SCRIPT
}

# ######################################################################################### DM #######################################

MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-31_16-49-37-42fac09f-f7a9-4466-a36c-f636e0886d9a/UAGL.pth"
DC=128
ARCH=IHN
submit_job_single $DC $ARCH $MODEL 3
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-31_17-20-32-f938fbaf-54ea-4d53-a045-d7d4a66a72b6/UAGL.pth"
DC=256
ARCH=IHN
submit_job_single $DC $ARCH $MODEL 3
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-31_18-41-34-cba6c674-e614-4550-8566-a24fd8248b2a/UAGL.pth"
DC=512
ARCH=IHN
submit_job_single $DC $ARCH $MODEL 3

MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-31_16-25-36-30223657-d3df-406f-994e-3b29e5095620/UAGL.pth"
DC=128
ARCH=IHN
submit_job_single $DC $ARCH $MODEL 2
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-31_17-18-30-56d89300-db97-4f4f-b2cd-3804d2c715bd/UAGL.pth"
DC=256
ARCH=IHN
submit_job_single $DC $ARCH $MODEL 2
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-14_07-03-25-2607ed33-70e1-4c35-8adf-7b61d32817aa/UAGL.pth"
DC=512
ARCH=IHN
submit_job_single $DC $ARCH $MODEL 2

# ######################################################################################### DE #######################################

MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_18-44-07-97a33213-80a2-4f50-9d85-9ad04d7df728/RHWF.pth"
DC=512
EN="./local_pipeline/ensembles/ensemble_512_IHN.txt"
ARCH=IHN
submit_job_ensemble $DC $EN $ARCH $MODEL 3
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_18-44-07-cfb3bb2c-e987-4c17-bdc0-731bc776dcdd/RHWF.pth"
DC=256
EN="./local_pipeline/ensembles/ensemble_256_IHN.txt"
ARCH=IHN
submit_job_ensemble $DC $EN $ARCH $MODEL 3
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-50-42-ed74e93e-0c1a-4926-9d12-21aa8c257e12/RHWF.pth"
DC=128
EN="./local_pipeline/ensembles/ensemble_128_IHN.txt"
ARCH=IHN
submit_job_ensemble $DC $EN $ARCH $MODEL 3

MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-04-27_00-45-46-07ebf9cb-0452-4163-a883-ef4a57b7a5a8/UAGL.pth"
DC=512
EN="./local_pipeline/ensembles/ensemble_512_STHN.txt"
ARCH=IHN
submit_job_ensemble $DC $EN $ARCH $MODEL 2
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-30_14-15-37-b45bf3b1-0e14-490e-b525-d2e9837838f8/UAGL.pth"
DC=256
EN="./local_pipeline/ensembles/ensemble_256_STHN.txt"
ARCH=IHN
submit_job_ensemble $DC $EN $ARCH $MODEL 2
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-30_15-38-56-174bfc03-f499-45c1-86cf-b8ddeaaa6ec5/UAGL.pth"
DC=128
EN="./local_pipeline/ensembles/ensemble_128_STHN.txt"
ARCH=IHN
submit_job_ensemble $DC $EN $ARCH $MODEL 2

# ######################################################################################### CROPTTA DE #######################################

# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-50-42-ed74e93e-0c1a-4926-9d12-21aa8c257e12/RHWF.pth"
ST=random
SHIFT=32
ARCH=IHN
EN="./local_pipeline/ensembles/ensemble_128_IHN.txt"
COM=max
DC=128
submit_job_crop_ensemble $DC $ST $SHIFT $ARCH $EN $COM $MODEL 3
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-04_12-39-21-59f0af94-1d92-4db6-b38c-739ffc7e522b/UAGL.pth"
ST=random
SHIFT=32
ARCH=IHN
EN="./local_pipeline/ensembles/ensemble_256_IHN.txt"
COM=max
DC=256
submit_job_crop_ensemble $DC $ST $SHIFT $ARCH $EN $COM $MODEL 3
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-04_02-58-47-d311243b-f6ed-4e5e-a317-a81308739cd0/UAGL.pth"
ST=random
SHIFT=32
ARCH=IHN
EN="./local_pipeline/ensembles/ensemble_512_IHN.txt"
COM=max
DC=512
submit_job_crop_ensemble $DC $ST $SHIFT $ARCH $EN $COM $MODEL 3

# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-02_20-47-51-5147b2ac-e11b-42e3-b8e7-4050501b3877/UAGL.pth"
ST=random
SHIFT=32
ARCH=IHN
EN="./local_pipeline/ensembles/ensemble_128_STHN.txt"
COM=max
DC=128
submit_job_crop_ensemble $DC $ST $SHIFT $ARCH $EN $COM $MODEL 2
# # w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-02_21-03-28-0969ae54-ba6a-4bcf-84da-1d46b7594784/UAGL.pth"
ST=random
SHIFT=32
ARCH=IHN
EN="./local_pipeline/ensembles/ensemble_256_STHN.txt"
COM=max
DC=256
submit_job_crop_ensemble $DC $ST $SHIFT $ARCH $EN $COM $MODEL 2
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-57-59-16ed57bd-e7c3-4575-8421-d180efbbff36/UAGL.pth"
ST=random
SHIFT=32
ARCH=IHN
EN="./local_pipeline/ensembles/ensemble_512_STHN.txt"
COM=max
DC=512
submit_job_crop_ensemble $DC $ST $SHIFT $ARCH $EN $COM $MODEL 2

# ######################################################################################### CROPTTA #######################################

# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_larger_ori_train-2024-02-18_20-50-42-ed74e93e-0c1a-4926-9d12-21aa8c257e12/RHWF.pth"
ST=random
SHIFT=32
ARCH=IHN
DC=128
submit_job_crop $DC $ST $SHIFT $ARCH $MODEL 3
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-04_15-04-28-5055f8ae-86a8-4b8c-84a7-6d7ccd74644a/UAGL.pth"
ST=random
SHIFT=32
ARCH=IHN
DC=256
submit_job_crop $DC $ST $SHIFT $ARCH $MODEL 3
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-04_02-58-47-d311243b-f6ed-4e5e-a317-a81308739cd0/UAGL.pth"
ST=random
SHIFT=32
ARCH=IHN
DC=512
submit_job_crop $DC $ST $SHIFT $ARCH $MODEL 3

# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-02_20-47-51-5147b2ac-e11b-42e3-b8e7-4050501b3877/UAGL.pth"
ST=random
SHIFT=32
ARCH=IHN
DC=128
submit_job_crop $DC $ST $SHIFT $ARCH $MODEL 2
# # w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-02_21-03-28-0969ae54-ba6a-4bcf-84da-1d46b7594784/UAGL.pth"
ST=random
SHIFT=32
ARCH=IHN
DC=256
submit_job_crop $DC $ST $SHIFT $ARCH $MODEL 2
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-57-59-16ed57bd-e7c3-4575-8421-d180efbbff36/UAGL.pth"
ST=random
SHIFT=32
ARCH=IHN
DC=512
submit_job_crop $DC $ST $SHIFT $ARCH $MODEL 2