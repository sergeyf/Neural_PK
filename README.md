# Tutorial on the Neural PK Model

From the paper Neural-ODE for pharmacokinetics modeling and its advantage to alternative machine learning models in predicting new dosing regimens
by Lu et al. 

The original code for this paper and the example data were shared here: https://github.com/jameslu01/Neural_PK

This repository refactors the original code and puts the entirety of the functionality in a single Jupyter notebook,
accompanied by various scripts that contain helpful utilities and wrappers. A few small bugs have been fixed

## Installation

Installation is easiest with Anaconda. Once you have it installed, execute the following in the Anaconda prompt:
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create --name neuralpk python=3.7.5
conda activate neuralpk
conda install pytorch==1.5 scipy==1.7.3 torchdiffeq==0.2.2 pandas==1.3.5 scikit-learn==1.0.2 jupyterlab tqdm 
``` 

If you have a GPU, note your CUDA version and install the dependencies with the CUDA toolkit. For example,
if you have CUDA 10.2, you can install the dependencies with the following command:
```
conda install pytorch==1.5 scipy==1.7.3 torchdiffeq==0.2.2 pandas==1.3.5 scikit-learn==1.0.2 jupyterlab tqdm cudatoolkit=10.2 -c pytorch
```
Only CUDA versions 10.2 and 11.3 are supported at the time of this writing. Please see https://pytorch.org/get-started/locally/
for more details.

## How to Use
Just run through `main.ipynb`. A lot of the code is in other files, which have been documented therein.