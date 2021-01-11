# Higher-Order Certification for Randomized Smoothing

This repository contains the code and models necessary to replicate the results of our recent paper:

**Higher-Order Certification for Randomized Smoothing** <br>
*Jeet Mohapatra, Irene Ko, Lily Weng, Sijia Liu, Pin-Yu Chen, Luca Daniel* <br>
Paper: https://arxiv.org/abs/2010.06651 <br>

## Overview of the Repository

Our code is based on the open source code of [Cohen et al (2019)](https://github.com/locuslab/smoothing). Let us dive into the files in [code/](code):

1. `train.py`: the original training code of [Cohen et al (2019)](https://github.com/locuslab/smoothing) using Gaussian noise data augmentation.
2. `certify.py`: Given a pretrained smoothed classifier, returns a certified first-order information based $\ell_1$-radius, $\ell_2$-radius, $\ell_\infty$-radus and subspace $\ell_2$-radius (for each of the R,G,B color channels) for each data point in a given dataset using the algorithm from the paper.
3. `predict.py`: Given a pretrained smoothed classifier, predicts the class of each data point in a given dataset.
4. `architectures.py`: an entry point for specifying which model architecture to use per dataset (Resnet-50 for ImageNet, Resnet-110 for CIFAR-10).
6. `numerical.py` : Provides for the code for numerically solving the set of integral equation constraints derived from the zeroth and the first-order conditions.

## Getting started

1.  Clone this repository: `git clone git@github.com:JeetMo/HigherOrderCertification.git`

2.  Install the dependencies:  
```
conda create -n smoothing
conda activate smoothing
# below is for linux, with CUDA 10; see https://pytorch.org/ for the correct command for your system
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch 
conda install scipy pandas statsmodels matplotlib seaborn
pip install setGPU
```
3.  Download the trained models from [here](https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view?usp=sharing).
4. If you want to run ImageNet experiments, obtain a copy of ImageNet and preprocess the `val` directory to look
like the `train` directory by running [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).
Finally, set the environment variable `IMAGENET_DIR` to the directory where ImageNet is located.

5. Now you can simulataneously certify the $\ell_1, \ell_2, \ell_\infty,$ and subspace $\ell_2$ robustness of one of the pretrained CIFAR-10 models
on the CIFAR test set by running
```
model="models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar"
output="???"
python code/certify.py cifar10 $model 0.25 $output --skip 20 --batch 400
```
where `???` is your desired output file.