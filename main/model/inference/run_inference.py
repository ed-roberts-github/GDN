from astropy.io import fits
import regions
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from regions import RegionVisual


import os
from tqdm import tqdm
import pathlib 
import csv
import sys

import pandas as pd
import numpy as np
from scipy.spatial import distance

import onnxruntime

from .infer_functions import (
    get_anchors,
    sigmoid,
    weighted_average_suppression,
    infer,
    write_output_csv,
    write_regions,
)

if __name__ == "__main__":
    config_infer = {

        ######### Change these to be the correct path on your computer #########
        "fits_path": "/Users/edroberts/Desktop/im_gen/training_data/testing/ceers/fits/",
        "onnx_path": "/Users/edroberts/Desktop/im_gen/StaryNight/onnx/model-195.onnx",
        "regions_path": "/Users/edroberts/Desktop/im_gen/training_data/testing/general_metrics/to_use/",
        "predictions_path": "/Users/edroberts/Desktop/im_gen/training_data/testing/ceers/predictions/",
        "image_name": "x7259y1686.fits", # change number here to view different images
        ########################################################################
    

        "image_size": 224,
        "feature_size": 28,
        "c_cutoff": 0.8,
        "r_suppression": 7.5,
        "r_tp": 8,
    }

    ########################### For one image ######################
    # Get the command-line fits name input
    args = sys.argv
    if len(args) == 2:
        assert (os.path.splitext(args[1])[1] == '.fits')
        config_infer['image_name'] = args[1]

    elif len(args) > 2:
        raise Exception("Too many command line inputs - only expect one: filename.fits")

    ort_session = onnxruntime.InferenceSession(config_infer["onnx_path"])
    input_name = ort_session.get_inputs()[0].name

    # # Open and standardise image
    # with fits.open(config_infer["fits_path"] + config_infer["image_name"]) as hdul:
    #     test_image = np.array(hdul[1].data).astype(np.float32)
    # test_image = np.expand_dims(test_image, 0)
    # test_image = np.expand_dims(test_image, 0) / 242.06  
    # ort_inputs = {input_name: test_image}

    # Open and standardise image
    with fits.open(config_infer["fits_path"] + config_infer["image_name"]) as hdul:
        test_image = np.array(hdul[0].data).astype(np.float32)
    test_image = np.expand_dims(test_image, 0)
    test_image = np.expand_dims(test_image, 0) / 1.3538227 #7.895 # Normalising real images
    ort_inputs = {input_name: test_image}

    # returns a list: first element is the confidence logits (pre-sigmoid), second is regression values
    ort_outs = ort_session.run(None, ort_inputs)

    # Converts the prediction labels to real x/y
    confidence, pixel_xy = infer(
        config_infer, ort_outs[0], ort_outs[1], get_anchors(config_infer)
    )
    
    print(f"Confidence values: \n {confidence} \n")
    print(f"DS9 pixel (x,y): \n {pixel_xy[0]} \n")

    
    regions_filename = config_infer['regions_path']+ os.path.splitext(config_infer['image_name'])[0] + '.reg'
    fits_filename = config_infer["fits_path"]+config_infer["image_name"]
    csv_filename = config_infer['predictions_path'] + os.path.splitext(config_infer['image_name'])[0]+ '.csv'

    write_regions(confidence[0], pixel_xy[0], regions_filename, fits_filename)
    write_output_csv(confidence[0], pixel_xy[0], csv_filename)
    ##############################################################
         