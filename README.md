# FBA-Net

This repository contains the code implementation for the paper "FBA-Net: Foreground and Background Aware Contrastive Learning for Semi-Supervised Atrium Segmentation".

## Requirements

List of dependencies required to run the code.

- python
- numpy
- torch
- h5py
- nibabel
- scipy
- skimage
- tqdm
- medpy

These dependencies can be installed using the following command:

    pip install -r requirements.txt

## Dataset

The dataset used for this project can be downloaded from the following links:

- https://www.cardiacatlas.org/atriaseg2018-challenge/
- https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT

## How to train

1. Clone the repository
2. Install the required packages using the command mentioned above
3. Download the dataset from the link provided above and extract it to the data/ directory
4. Run the following command to train the model:
   ```bash
   # e.g. train FBA-Net+ with 20% labeled data on the LA dataset.
   python ./code/train.py --dataset_name LA --model fbapnet --labelnum 16 --gpu 0 --dim 128 --exp FBA_net_plus
   ```
5. To evaluate the model, run the following command:
   ```bash
   python ./code/test.py --dataset_name LA --model fbapnet --labelnum 16 --gpu 0 --dim 128 --exp FBA_net_plus
   ```

## Results

The following tables show the results obtained in our experiments on the LA dataset:
| Method | # Scans Used | Labeled | Dice (\%) | Jaccard (\%) | ASD (voxel) | 95HD (voxel) |
|-------------------|----------------|------------|------------|-----------------|----------------|----------------|
| V-Net | 80 | 0 | 91.62 | 84.60 | 1.64 | 5.40 |
| V-Net | 16 | 0 | 86.96 | 77.31 | 3.22 | 11.85 |
| V-Net | 8 | 0 |78.57 |66.96 |6.07 |21.2|
|-----------------|--------------|----------|----------|---------------|--------------|--------------|
| SSASSNet | 16 | 64 | 89.23 | 80.82 | 3.15 | 8.81 |
| DTC | 16 | 64 | 89.39 | 80.95 | 2.16 | 7.44 |
| CVRL | 16 | 64 | 90.15 | 82.42 | 2.01 | 6.91 |
| MC-NET | 16 | 64 | 90.11 | 82.09 | 2.02 | 7.86 |
| MC-NET+ | 16 | 64 | 91.09 | 83.70 | 1.71 | 5.77 |
| **FBA-Net** | 16 | 64 | 89.77 | 81.57 | 1.62 | 7.43 |
| **FBA-Net+** | 16 | 64 | **91.31** | **84.07** | **1.52** | **5.44** |
|-----------------|--------------|----------|----------|---------------|--------------|--------------|
| SSASSNet | 8 | 72 | 85.81 | 76.99 | 4.04 | 15.54 |
| DTC | 8 | 72 | 87.91 | 78.98 | 2.96 | 8.44 |
| CVRL | 8 | 72 | 88.06 | 78.53 | 3.11 | 8.98 |
| MC-NET | 8 | 72 | 87.92 | 77.59 | 2.64 | 13.86 |
| MC-NET+ | 8 | 72 | 88.39 | 79.10 | 1.99 | **8.11** |
| **FBA-Net** | 8 | 72 | 87.52 | 78.04 | 2.19 | 9.06 |
| **FBA-Net+** | 8 | 72 | **88.69** | **79.84** | **1.92** | 8.84 |

### Acknowledgment

We have adapted our code from [UAMT](https://github.com/yulequan/UA-MT), [SSASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [MC-Net](https://github.com/ycwu1997/MC-Net), and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). We would like to thank the authors of these works for their valuable contributions, and we hope that our model can also contribute to further research in this field.
