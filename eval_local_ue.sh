#!/bin/bash

submit_job_ensemble() {
    local DC=$1
    local EN=$2
    local ARCH=$3
    local MODEL=$4
    local METHOD=$5

    if [ "$METHOD" = 1 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_e_val.sbatch"
    elif [ "$METHOD" = 2 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_e.sbatch"
    elif [ "$METHOD" = 3 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_esthn.sbatch"
    elif [ "$METHOD" = 4 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_eone.sbatch"
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
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_d_val.sbatch"
    elif [ "$METHOD" = 2 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_d.sbatch"
    elif [ "$METHOD" = 3 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_dsthn.sbatch"
    elif [ "$METHOD" = 4 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_done.sbatch"
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
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_c_val.sbatch"
    elif [ "$METHOD" = 2 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_c.sbatch"
    elif [ "$METHOD" = 3 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_csthn.sbatch"
    elif [ "$METHOD" = 4 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_cone.sbatch"
    else
        exit 0
    fi

    sbatch --export=ALL,DC=$DC,ST=$ST,SHIFT=$SHIFT,USEED=0,STDM=all,AGG=zero,CS=-1,OUT=$OUT,OUTM=$OUTM,DIS=$DIS,ARCH=$ARCH,MODEL=$MODEL $EVAL_SCRIPT
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
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_se_val.sbatch"
    elif [ "$METHOD" = 2 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_se.sbatch"
    elif [ "$METHOD" = 3 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_sesthn.sbatch"
    elif [ "$METHOD" = 4 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_ue1_seone.sbatch"
    else
        exit 0
    fi

    sbatch --export=ALL,DC=$DC,ST=$ST,SHIFT=$SHIFT,USEED=0,STDM=all,AGG=zero,CS=-1,OUT=$OUT,OUTM=$OUTM,DIS=$DIS,ARCH=$ARCH,EN=$EN,COM=$COM,MODEL=$MODEL $EVAL_SCRIPT
}

submit_job_baseline() {
    local DC=$1
    local MODEL=$2
    local METHOD=$3

    if [ "$METHOD" = 1 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2_val.sbatch"
    elif [ "$METHOD" = 2 ]
    then
        local EVAL_SCRIPT="scripts/local_largest_1536/eval_local_sparse_512_extended_2.sbatch"
    else
        exit 0
    fi

    sbatch --export=ALL,DC=$DC,MODEL=$MODEL $BASELINE_SCRIPT
}

# ################################################################################################################################################################################Train Methods

# TTA no crop training
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-04-27_00-45-46-07ebf9cb-0452-4163-a883-ef4a57b7a5a8"
ST=random
SHIFT=32
ARCH=IHN
submit_job_crop $ST $SHIFT $ARCH $MODEL

# #######################################################################################################################################################################################

# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-57-59-bb6fdc0c-65cb-457a-a562-3a81a63db1d9"
ST=grid
SHIFT=64
ARCH=IHN
submit_job_crop $ST $SHIFT $ARCH $MODEL

# w/ Crop Training - scratch - 5 - mr
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-58-04-5dde4e75-43c9-4ab4-845c-a3947b63abd3"
ST=random
SHIFT=64
ARCH=IHN
submit_job_crop $ST $SHIFT $ARCH $MODEL

# # ################################################################################################################################################################################Train Methods

# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-57-59-6ab9f306-76bc-4c31-bea2-2587eb306447"
ST=grid
SHIFT=32
ARCH=IHN
submit_job_crop $ST $SHIFT $ARCH $MODEL

# w/ Crop Training - scratch - 5 - mr
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-57-59-16ed57bd-e7c3-4575-8421-d180efbbff36"
ST=random
SHIFT=32
ARCH=IHN
submit_job_crop $ST $SHIFT $ARCH $MODEL

# w/ Crop Training - scratch - 3 - mr
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_21-31-21-0e5a3893-b77e-4493-a1af-14809a6a5efb"
ST=random
SHIFT=32
ARCH=IHN
submit_job_crop $ST $SHIFT $ARCH $MODEL

# # ################################################################################################################################################################################Train Methods

# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-57-59-4ebb9b13-1d6f-4404-9610-245b9c216c74"
ST=grid
SHIFT=16
ARCH=IHN
submit_job_crop $ST $SHIFT $ARCH $MODEL

# w/ Crop Training - scratch - 5 - mr
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-57-59-1dfa2add-6c74-47c4-a4d4-90dea5d70501"
ST=random
SHIFT=16
ARCH=IHN
submit_job_crop $ST $SHIFT $ARCH $MODEL

# ######################################################################################### DM #######################################

MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-04_23-47-31-1c8b9a7f-8ff4-4916-b38f-a4f17413d0ce/UAGL.pth"
DC=512
ARCH=DHN
submit_job_single $DC $ARCH $MODEL 4
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_15-36-16-14323b3d-a6f2-4898-bb6b-e06eb41a5c1f/UAGL.pth"
DC=256
ARCH=DHN
submit_job_single $DC $ARCH $MODEL 4
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_10-37-23-dbc3cb9d-68ad-4480-8ca7-fd31d19add79/UAGL.pth"
DC=128
ARCH=DHN
submit_job_single $DC $ARCH $MODEL 4

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

MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-19_12-34-56-acfb33d8-497c-4396-98d7-7485fdc30a13"
DC=512
EN="./local_pipeline/ensembles/ensemble_512_DHN.txt"
ARCH=DHN
submit_job_ensemble $DC $EN $ARCH $MODEL 4
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-19_14-39-21-5d862420-4465-44c5-95c3-9ddd8e796d51"
DC=256
EN="./local_pipeline/ensembles/ensemble_256_DHN.txt"
ARCH=DHN
submit_job_ensemble $DC $EN $ARCH $MODEL 4
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-05-28_18-02-55-fd0e2a0f-93fe-43b2-b375-5a63b549d55e"
DC=128
EN="./local_pipeline/ensembles/ensemble_128_DHN.txt"
ARCH=DHN
submit_job_ensemble $DC $EN $ARCH $MODEL 4

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
# submit_job_ensemble $DC $EN $ARCH $MODEL 2
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

# ######################################################################################### CROPTTA DE val #######################################

# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-57-59-16ed57bd-e7c3-4575-8421-d180efbbff36"
ST=random
SHIFT=32
ARCH=IHN
EN="./local_pipeline/ensembles/ensemble_512_STHN.txt"
COM=min
submit_job_crop_ensemble $ST $SHIFT $ARCH $EN $COM $MODEL
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-57-59-16ed57bd-e7c3-4575-8421-d180efbbff36"
ST=random
SHIFT=32
ARCH=IHN
EN="./local_pipeline/ensembles/ensemble_512_STHN.txt"
COM=add
submit_job_crop_ensemble $ST $SHIFT $ARCH $EN $COM $MODEL
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-01_14-57-59-16ed57bd-e7c3-4575-8421-d180efbbff36"
ST=random
SHIFT=32
ARCH=IHN
EN="./local_pipeline/ensembles/ensemble_512_STHN.txt"
COM=max
submit_job_crop_ensemble $ST $SHIFT $ARCH $EN $COM $MODEL

# ######################################################################################### CROPTTA DE #######################################

# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-04_16-28-22-97909f35-efa0-4568-9ce1-97dbd2a0ffbe/UAGL.pth"
ST=random
SHIFT=32
ARCH=DHN
EN="./local_pipeline/ensembles/ensemble_128_DHN.txt"
COM=max
DC=128
submit_job_crop_ensemble $DC $ST $SHIFT $ARCH $EN $COM $MODEL 4
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-02_20-47-48-7e50d890-99e2-40e9-b429-22d591eb30ce/UAGL.pth"
ST=random
SHIFT=32
ARCH=DHN
EN="./local_pipeline/ensembles/ensemble_256_DHN.txt"
COM=max
DC=256
submit_job_crop_ensemble $DC $ST $SHIFT $ARCH $EN $COM $MODEL 4
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-02_21-08-28-297e8fe7-8688-4d2c-bf75-ad6402aada9d/UAGL.pth"
ST=random
SHIFT=32
ARCH=DHN
EN="./local_pipeline/ensembles/ensemble_512_DHN.txt"
COM=max
DC=512
submit_job_crop_ensemble $DC $ST $SHIFT $ARCH $EN $COM $MODEL 4

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
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-04_16-28-22-97909f35-efa0-4568-9ce1-97dbd2a0ffbe/UAGL.pth"
ST=random
SHIFT=32
ARCH=DHN
DC=128
submit_job_crop $DC $ST $SHIFT $ARCH $MODEL 4
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-02_20-47-48-7e50d890-99e2-40e9-b429-22d591eb30ce/UAGL.pth"
ST=random
SHIFT=32
ARCH=DHN
DC=256
submit_job_crop $DC $ST $SHIFT $ARCH $MODEL 4
# w/ Crop Training - scratch - 5 - m
MODEL="satellite_0_thermalmapping_135_nocontrast_dense_exclusion_largest_ori_train-2024-06-02_21-08-28-297e8fe7-8688-4d2c-bf75-ad6402aada9d/UAGL.pth"
ST=random
SHIFT=32
ARCH=DHN
DC=512
submit_job_crop $DC $ST $SHIFT $ARCH $MODEL 4

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