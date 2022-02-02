# Tutorial on the Neural PK Model

From the paper "Neural-ODE for pharmacokinetics modeling and its advantage to alternative machine learning models in predicting new dosing regimens"
by Lu et al. 

The original code for this paper and the example data were shared here: https://github.com/jameslu01/Neural_PK

This repository refactors the original code and puts the functionality in a single Jupyter notebook,
accompanied by various scripts that contain helpful utilities and wrappers. A few small bugs have been fixed.

Workflow             |  Model Schemata
:-------------------------:|:-------------------------:
![](figures/figure1.png)  |  ![](figures/figure2.png)


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
Just run through `walkthrough.ipynb`, which is heavily documented.
A lot of the functionality that is called from `walkthrough.ipynb` code is in other files, which have been documented also.
The actual training is done for 30 epochs and 25 different times because there are 5 training/test splits
and for each split 5 model variants are trained to be used in model averaging. We ran these models on a GPU
and it took multiple days to finish training. The checkpoints and evaluation results are stored in folders
`fold_1`, `fold_2`, etc. You can skip the training cell in `walkthrough.ipynb` and the rest of the cells 
should still execute.

## Manifest
- `ExampleData` - This folder contains the example data used in the paper, which was provided by the authors.
- `figures` - This folder contains some helpful figures that accompanied the original repository.
- `fold_1`, `fold_2`, etc. - These folders contain the checkpoints and evaluation results for the training/test splits.
- `data_parse.py` - Classes and functions to process clinical PK data from trastuzumab emtansine (T-DM1).
- `data_split.py` - Utility functions for data splitting and augmentation.
- `model.py` - The PyTorch model definitions.
- `README.md` - This file.
- `utils.py` - Miscellaneous utility.
- `walkthrough.ipynb` - The Jupyter notebook that contains the main code path.