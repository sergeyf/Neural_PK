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

If you have a GPU, you'll need to install CUDA 10.2 or 11.3. Then, install `pytorch` with `pip` as per instructions here: `https://pytorch.org/get-started/locally/`
which will be different for your specific system and CUDA version. Do *not* use `conda`. 
Also, the GPU version of `pytorch` 1.5.1 is not available any more, so you can install the latest `1.x` version.
After you install `pytorch`, you can install the rest of the dependencies with `pip` also:
```
pip install scipy torchdiffeq pandas scikit-learn jupyterlab tqdm 
```

## How to Use
Just run through `main.ipynb`. A lot of the code is in other files, which have been documented therein.