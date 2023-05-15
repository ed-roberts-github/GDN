# Galaxy Detection Network

All code used to build the galaxy detection network model is contained within this directory. 
(NOTE: this is a different git repo to the repo used in development)

## Dirs
/main/ contains all the coded used to build, train and test the model. A sepearte README describes its contents.

/image_generation contains all the code used for image geeration. A sepearte README describes its contents.

/data/ contains data used for training and testing. (Note due to the size of the data all this data is exlcuded from actualky being in the file)

## Requirements

Different environments were used for each task. These are simply installed using pip...

For generation:
- numpy
- scipy
- powerlaw
- astropy
- galsim

For main:
- torch
- onnx
- onnxruntime
- tqdm
- numpy
- h5py
- timm
- torchmetrics
- wandb
- torchvision
- numpy
- pytorch_lightning (they are undergoing the process of updating this so may be just lightning now)
- sep
- scipy
- astropy
- matplotlib
- pandas
- regions
