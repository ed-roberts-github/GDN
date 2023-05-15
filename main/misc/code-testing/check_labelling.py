import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pandas as pd
from tqdm import tqdm
import h5py
from .functions import get_anchors, get_anchor_labels 

test_config = {
    "image_size": 224,
    "feature_size": 14,
    "r_far": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),
    "r_near": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),
    "batch_size": 128,
}

prepros_config = {
    "path_to_fits": "/Users/edroberts/Desktop/im_gen/training_data/train/train/fits",
    "path_to_csv": "/Users/edroberts/Desktop/im_gen/training_data/train/train/csv",
    "path_to_main": "/Users/edroberts/Desktop/im_gen/training_data/train/train",
    "image_name": "train",
    "hdf5_name": "train_dataset14",
    "expected_img_number": 29989,
}

# os.chdir(prepros_config["path_to_fits"])
# with fits.open("simple_00021.fits") as hdul:
#             #hdul.info()
#             data_images = np.array(hdul[1].data)

conf_labels = np.full((1, 3, test_config['feature_size'],
                        test_config['feature_size']), -1, dtype=np.float64)

reg_labels = np.full((1, 3, 2, test_config['feature_size'],
                        test_config['feature_size']), -1, dtype=np.float64)

print(reg_labels.shape)

anchors = get_anchors(test_config)
os.chdir(prepros_config["path_to_csv"])

df_center = pd.read_csv("train_00000.csv", sep=',', header=None)
np_center = df_center.values
np_center[[0,1]] = np_center[[1,0]] # Swapping from y,x to x,y
        
# Remember .T for np_centers file to transpose from columns to rows
conf_labels[0], reg_labels[0] = get_anchor_labels(anchors, np_center.T, test_config)

# print(anchors)

# print(conf_labels)

print(conf_labels[0][0])

print(reg_labels[0][0])
