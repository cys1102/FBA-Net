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
   ```console
   # e.g. train FBA-Net+ with 20% labeled data on the LA dataset.
   python ./code/train.py --dataset_name LA --model fbapnet --labelnum 16 --gpu 0 --dim 128 --exp FBA_net_plus
   ```
5. To evaluate the model, run the following command:
   ```console
   python ./code/test.py --dataset_name LA --model fbapnet --labelnum 16 --gpu 0 --dim 128 --exp FBA_net_plus
   ```

## Results

The following tables show the results obtained in our experiments on the LA dataset:
|Method||||
Model Accuracy
Model 1 0.85
Model 2 0.92
Model 3 0.96

### Acknowledgment

We have adapted our code from [UAMT](https://github.com/yulequan/UA-MT), [SSASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [MC-Net](https://github.com/ycwu1997/MC-Net), and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). We would like to thank the authors of these works for their valuable contributions, and we hope that our model can also contribute to further research in this field.
