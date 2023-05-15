### Program to calculate the purity and completeness for different SNR ###

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
    write_snr_csv,
)

if __name__ == "__main__":
    config_infer = {
        ######### Change these to be the correct path on your computer #########
        "main_path": "/Users/edroberts/Desktop/im_gen/training_data/testing/snr/",
        "onnx_path": "/Users/edroberts/Desktop/im_gen/StaryNight/onnx/model-195.onnx",
        "basename": "snr",
        ########################################################################
        "image_size": 224,
        "feature_size": 28,
        "c_cutoff": 0.6,
        "r_suppression": 7.5,
        "r_tp": 8,
    }

    # suppress numpy overflow warnings from exp()
    warnings.filterwarnings("ignore")

    snr_list = [
        "1","2","3","4","5","6","7","8","9","10",
        "11","12","13","14","15","16","17","18",
        "19","20","25","30","35","40","45","50",
    ]

    purity_list = []
    purity_err_list = []
    completeness_list = []
    completeness_err_list = []

    for snr in snr_list:
        fits_names = []
        csv_names = []
        file_numbers = find_files(
            100,
            config_infer["main_path"] + "/snr" + snr + "/fits",
            config_infer["basename"] + snr,
        )
        for num in file_numbers[0]:
            fits_names.append(
                config_infer["main_path"]
                + "snr"
                + snr
                + "/fits/"
                + config_infer["basename"]
                + snr
                + "_"
                + num
                + ".fits"
            )
            csv_names.append(
                config_infer["main_path"]
                + "snr"
                + snr
                + "/csv/"
                + config_infer["basename"]
                + snr
                + "_"
                + num
                + ".csv"
            )

        p, c = metric_loop(config_infer, fits_names, csv_names, config_infer["r_tp"])

        # print(np.mean(p))
        # print(np.mean(c))

        purity_list.append(np.mean(p))
        purity_err_list.append(np.std(p))

        completeness_list.append(np.mean(c))
        completeness_err_list.append(np.std(c))

    print(purity_list)
    print(completeness_list)

    write_snr_csv(
        config_infer,
        'SNR_sweepv3', 
        snr_list, 
        purity_list, 
        purity_err_list, 
        completeness_list, 
        completeness_err_list
    )

    # plt.plot([eval(i) for i in snr_list],completeness_list)
