#!/bin/bash

# 128
python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_largest_ori_train --identity --lev0 --database_size 1536 --corr_level 4 --generate_test_pairs --val_positive_dist_threshold 128

python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_largest_ori_train --identity --database_size 1536 --test --val_positive_dist_threshold 128

# 256
python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_largest_ori_train --identity --lev0 --database_size 1536 --corr_level 4 --generate_test_pairs --val_positive_dist_threshold 256

python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_largest_ori_train --identity --database_size 1536 --test --val_positive_dist_threshold 256

# 512
python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_largest_ori_train --identity --lev0 --database_size 1536 --corr_level 4 --generate_test_pairs --val_positive_dist_threshold 512

python3 ./local_pipeline/myevaluate.py --dataset_name satellite_0_thermalmapping_135_largest_ori_train --identity --lev0 --database_size 1536 --corr_level 4 --test --generate_test_pairs --val_positive_dist_threshold 512



