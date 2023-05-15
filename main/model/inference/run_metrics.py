from astropy.io import fits
import regions
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from regions import RegionVisual
# from astropy import units as u # TODO should i use the units module?

import os
from tqdm import tqdm
import pathlib 
import csv
import sys
import warnings

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
    find_files,
    scores,
    purity,
    completeness,
    metric_loop,
    write_metrics_csv,
)

if __name__ == "__main__":
    config_infer = {

        ######### Change these to be the correct path on your computer #########
        "main_path": "/Users/edroberts/Desktop/im_gen/training_data/testing/general_metrics/",
        "onnx_path": "/Users/edroberts/Desktop/im_gen/StaryNight/onnx/model_7x7c.onnx",
        "basename": 'testing',
        ########################################################################

        "image_size": 224,
        "feature_size": 7,
        "c_cutoff": 0.6,
        "r_suppression": 7.5,
        "r_tp": 8,
    }

    # suppress numpy overflow warnings from exp()
    warnings.filterwarnings('ignore')

    # Getting al file names
    fits_names = []
    csv_names = []
    file_numbers = find_files(1000, config_infer['main_path']+'/fits', config_infer['basename'])
    for num in file_numbers[0]:
        fits_names.append(config_infer["main_path"]+ "/fits/"+ config_infer['basename']+'_'+num+'.fits')
        csv_names.append(config_infer["main_path"]+ "/csv/"+config_infer['basename']+'_'+num+'.csv')

    r_tp_range = np.arange(1,33,1)
    purity_list = []
    p_err = []
    completeness_list =[]
    c_err = []
    
    for r in r_tp_range:
        p,c = metric_loop(config_infer, fits_names, csv_names, r)

        purity_list.append(np.mean(p))
        p_err.append(np.std(p))
        completeness_list.append(np.mean(c))
        c_err.append(np.std(c))

    
        print(f'r_tp: {r}')
        print(f'Mean purity: {np.mean(p)}')
        print(f'Mean completeness: {np.mean(c)}')


    print(purity_list)
    print(completeness_list)

    write_metrics_csv(config_infer,
                      "metrics_7x7v5",
                      list(r_tp_range), 
                      purity_list,
                      p_err,
                      completeness_list,
                      c_err
                    )
