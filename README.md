# UASTHN: Uncertainty-Aware Deep Homography Estimation for UAV Satellite-Thermal Geo-localization

[![arXiv](https://img.shields.io/badge/arXiv-2502.01035-B31B1B.svg)](https://arxiv.org/abs/2502.01035)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face%20Dataset-UASTHN-blue.svg)](https://huggingface.co/datasets/xjh19972/boson-nighttime/tree/main/satellite-thermal-dataset-v3)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face%20Model-UASTHN-blue.svg)](https://huggingface.co/xjh19972/UASTHN)
[![Model](https://img.shields.io/badge/Resources-UASTHN-green.svg)](https://drive.google.com/drive/folders/1cEH8vXzt0TMrJ6_SVaJwv4GxWYdWduZw?usp=sharing)

This is the official repository for [UASTHN: Uncertainty-Aware Deep Homography Estimation for UAV Satellite-Thermal Geo-localization](https://arxiv.org/abs/2502.01035).

Related works:  
* Long-range UAV Thermal Geo-localization with Satellite Imagery [[Project]](https://xjh19971.github.io/STGL)
* STHN: Deep Homography Estimation for UAV Thermal Geo-localization with Satellite Imagery [[Project]](https://xjh19971.github.io/STHN)

```
@INPROCEEDINGS{xiao2025uasthn,
  author={Xiao, Jiuhong and Loianno, Giuseppe},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={UASTHN: Uncertainty-Aware Deep Homography Estimation for UAV Satellite-Thermal Geo-Localization}, 
  year={2025},
  volume={},
  number={},
  pages={14066-14072},
  keywords={Location awareness;Uncertainty;Satellites;Measurement uncertainty;Estimation;Autonomous aerial vehicles;Thermal noise;Robustness;Noise measurement;Robotics and automation},
  doi={10.1109/ICRA55743.2025.11128423}}
```

## Conda Environment Setup
Our repository requires a conda environment. Relevant packages are listed in ``env.yml``. Run the following command to setup the conda environment.
```
conda env create -f env.yml
```

## Simple Inference Demo

To quickly get started with UASTHN, we provide a [`UASTHN_demo.py`](STHN_demo.py) script. This demo automatically downloads the model from Hugging Face and runs inference on a pair of sample images:

```bash
# From the UASTHN root directory
conda activate UASTHN
python UASTHN_demo.py
```

## Dataset
We modify the Boson-nighttime dataset from [STGL](https://github.com/arplaboratory/satellite-thermal-geo-localization/tree/main) and [STHN](https://github.com/arplaboratory/STHN) to use the entire satellite map instead of h5 files. Now you can use center coordinates and image width to crop the satellite images from the map so that the dataset size is reduced and the crop width is flexible. Thermal images are still in h5 files because we haven't found a way to stitch the generated thermal images into one map appropriately.

The raw resolution of the thermal images is 768x768, allowing for data augmentation (e.g., resizing and rotation) without introducing black padding. For training, we crop the central 512x512 region in the dataloader after augmentation.

Dataset link (122 GB): [Download](https://huggingface.co/datasets/xjh19972/boson-nighttime/tree/main/satellite-thermal-dataset-v3)

The ``datasets`` folder should be created in the root folder with the following structure.

```
UASTHN/datasets/
├── maps
│   └── satellite
|   |   └── 20201117_BingSatellite.png
├── satellite_0_satellite_0
│   └── train_database.h5
├── satellite_0_thermalmapping_135
│   ├── test_database.h5
│   ├── test_queries.h5
│   ├── train_database.h5
│   ├── train_queries.h5
│   ├── val_database.h5
│   └── val_queries.h5
├── satellite_0_thermalmapping_135_train
│   ├── extended_database.h5 -> ../satellite_0_satellite_0/train_database.h5
│   ├── extended_queries.h5
│   ├── test_database.h5 -> ../satellite_0_thermalmapping_135/test_database.h5
│   ├── test_queries.h5 -> ../satellite_0_thermalmapping_135/test_queries.h5
│   ├── train_database.h5 -> ../satellite_0_thermalmapping_135/train_database.h5
│   ├── train_queries.h5 -> ../satellite_0_thermalmapping_135/train_queries.h5
│   ├── val_database.h5 -> ../satellite_0_thermalmapping_135/val_database.h5
│   └── val_queries.h5 -> ../satellite_0_thermalmapping_135/val_queries.h5
```

## Training
You can find the single training and evaluation scripts in the ``scripts/local_largest_1536`` folder. The scripts are for slurm system to submit sbatch job. You can also directly run the script with bash.
For training, please refer to ``train_local_ue.sh`` and change necessary arguments, such as $MODEL_FOLDER.

After training, find your model folder in ``./logs/local_he/$dataset_name-$datetime-$uuid``.

## Evaluation
For evaluation, please refer to ``eval_local_ue.sh`` and change necessary arguments, such as $MODEL_FOLDER.

Find the test results in ``./test/local_he/$model_folder_name/``.  

## Image-matching Baselines
For training and evaluating the image-matching baselines (anyloc and STGL), please refer to ``scripts/global/`` for training and evaluation.

## Pretrained Models
Download pretrained models with CropTTA: [Download](https://drive.google.com/drive/folders/1cEH8vXzt0TMrJ6_SVaJwv4GxWYdWduZw?usp=sharing)

## Additional Details
<details>
  <summary>Train/Val/Test split</summary>
  Below is the visualization of the train-validation-test regions. The dataset includes thermal maps from six flights: three flights (conducted at 9 PM, 12 AM, and 2 AM) cover the upper region, and the other three flights (conducted at 10 PM, 1 AM, and 3 AM) cover the lower region. The lower region is further divided into training and validation subsets. The synthesized thermal images span a larger area (23,744m x 9,088m) but exclude the test region to assess generalization performance properly.

  **[2025/12] Update:** We have released an updated version of **satellite-thermal-dataset-v3**, which excludes the test region from the generated data to ensure full alignment with the original paper’s evaluation protocol. If you wish to conduct a **rigorous comparison with STHN**, please replace `extended_queries.h5` with `extended_queries_test_excluded.h5`.
    
  ![image](https://github.com/arplaboratory/STHN/assets/29690116/8e833ba9-644e-4446-b951-7b17a5e4316b)
  
</details>
<details>
  <summary>Architecture Details</summary>
  The feature extractor consists of multiple residual blocks with multi-layer CNN and instance normalization:  
  https://github.com/arplaboratory/STHN/blob/0ad04d7fb19ba369d24184cda80941640c618631/local_pipeline/extractor.py#L177
  The iterative updater is a multi-layer CNN with group normalization:  
  https://github.com/arplaboratory/STHN/blob/eed553fb45756ce5ea35418db77383732c444c42/local_pipeline/update.py#L299  
  The TGM is using the Pix2Pix paradigm:
  https://github.com/arplaboratory/STHN/blob/eed553fb45756ce5ea35418db77383732c444c42/global_pipeline/model/network.py#L273
  
</details>

<details>
  <summary>ROC curves and MACE histogram</summary>
  Here are the ROC curve and MACE histogram for different values of $D_C$. These results indicate that our uncertainty estimation method performs well when $D_C$ is large, and the model exhibits a long-tail error distribution.
  
  <img src="https://github.com/arplaboratory/UASTHN/blob/master/assets/grid_random_5_10_roc.png" width=40% height=40%><img src="https://github.com/arplaboratory/UASTHN/blob/master/assets/grid_random_5_10_hist.png" width=40% height=40%>
  
</details>

## Acknowledgement
Our implementation refers to the following repositories and appreciate their excellent work.

https://github.com/imdumpl78/IHN  
https://github.com/AnyLoc/AnyLoc  
https://github.com/gmberton/deep-visual-geo-localization-benchmark  
https://github.com/fungtion/DANN  
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
