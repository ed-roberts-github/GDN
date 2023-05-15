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

import pandas as pd
import numpy as np
from scipy.spatial import distance

import onnxruntime

def get_anchors(config):
    """
    Calculates the anchor origin positions for a given image.
    """
    step = config["image_size"] / config["feature_size"]
    halfstep = step * 0.5

    # Calculates the lattice of points
    output = np.full(
        (3, 2, config["feature_size"], config["feature_size"]), -1, dtype=np.float32
    )
    x = np.arange(halfstep, config["image_size"], step, dtype=np.float32)
    y = np.arange(halfstep, config["image_size"], step, dtype=np.float32)
    grid = np.transpose([np.tile(x, len(y)), y.repeat(len(x))]).reshape(
        config["feature_size"], config["feature_size"], 2
    )

    # Could be done quicker by vectorisation but only done once so not a performace issue
    for i in range(len(x)):
        for j in range(len(y)):
            for d in range(3):
                output[d][0][i][j] = grid[i][j][0]
                output[d][1][i][j] = grid[i][j][1]

    return output

def sigmoid(confidence_logits):
    return 1.0 / (1.0 + np.exp(-confidence_logits))


def NMS(out_confs, pred_x, pred_y, r_nms):
    """
    Performs Non-maximum suppression to remove duplicate
    points.

    Inputs are intended to be for a single image.

    Note: THIS IS A LEGACY FUNCTION! FOUND W-A-S worked
    BETTER!
    """
    NMS_conf = []
    NMS_x = []
    NMS_y = []
    order_args = list(np.flip(np.array(out_confs).argsort()))

    count = 0
    while order_args != []:
        NMS_conf.append(out_confs[order_args[0]])
        NMS_x.append(pred_x[order_args[0]])
        NMS_y.append(pred_y[order_args[0]])
        order_args.remove(order_args[0])

        to_remove = []
        for arg in order_args:
            distance = np.sqrt(
                (NMS_x[count] - pred_x[arg]) ** 2 + (NMS_y[count] - pred_y[arg]) ** 2
            )
            if distance <= r_nms:
                to_remove.append(arg)

        [order_args.remove(arg) for arg in to_remove]
        count += 1

    return NMS_conf, NMS_x, NMS_y


def weighted_average_suppression(out_confs, pred_x, pred_y, r_supression):
    """
    W-A-S:
    Removes duplicate predictions by performing
    a weighted average of the points.

    Inputs are intended to be for a single image.
    """
    predicted_xy = np.ndarray.tolist(np.column_stack((pred_x, pred_y)))
    suppressed_conf = []
    suppressed_xy = []

    while predicted_xy != []:
        distances = distance.cdist([predicted_xy[0]], predicted_xy)
        indices = [i for (i, dist) in enumerate(distances[0]) if dist < r_supression]

        if len(indices) == 1:
            suppressed_conf.append(out_confs[0])
            suppressed_xy.append(predicted_xy[0])

            del predicted_xy[0]
            del out_confs[0]

        elif len(indices) == 2:
            ave_xy = (
                np.array(predicted_xy[0]) * out_confs[0]
                + np.array(predicted_xy[indices[1]]) * out_confs[indices[1]]
            ) / (out_confs[0] + out_confs[indices[1]])

            suppressed_conf.append(np.mean((out_confs[0], out_confs[indices[1]])))
            suppressed_xy.append(np.ndarray.tolist(ave_xy))

            del predicted_xy[indices[1]]
            del out_confs[indices[1]]
            del predicted_xy[0]
            del out_confs[0]

        elif len(indices) == 3:
            ave_xy = (
                np.array(predicted_xy[0]) * out_confs[0]
                + np.array(predicted_xy[indices[1]]) * out_confs[indices[1]]
                + np.array(predicted_xy[indices[2]]) * out_confs[indices[2]]
            ) / (out_confs[0] + out_confs[indices[1]] + out_confs[indices[2]])

            suppressed_conf.append(np.mean((out_confs[0], out_confs[indices[1]])))
            suppressed_xy.append(np.ndarray.tolist(ave_xy))

            del predicted_xy[indices[2]]
            del out_confs[indices[2]]
            del predicted_xy[indices[1]]
            del out_confs[indices[1]]
            del predicted_xy[0]
            del out_confs[0]

        else:
            raise Exception(
                "Weighted average suppression: More than two prediction within suppression \
                range: implment higher averaging"
            )

    return suppressed_conf, suppressed_xy


def infer(config, out_conf, out_reg, points):
    step = config["image_size"] / config["feature_size"]
    c_cutoff = config["c_cutoff"]
    r_sup = config["r_suppression"]

    out_pixels = (out_reg * step) + points
    out_conf = sigmoid(out_conf)

    batch_conf = []
    batch_x = []
    batch_y = []

    for i in (range(out_conf.shape[0])):
        high_conf_idx = np.where(out_conf[i] > c_cutoff)
        conf_list = []
        x_list = []
        y_list = []

        for j in range(high_conf_idx[0].shape[0]):
            conf_list.append(
                out_conf[
                    i, high_conf_idx[0][j], high_conf_idx[1][j], high_conf_idx[2][j]
                ]
            )
            x_list.append(
                out_pixels[
                    i, high_conf_idx[0][j], 0, high_conf_idx[1][j], high_conf_idx[2][j]
                ]
            )
            y_list.append(
                out_pixels[
                    i, high_conf_idx[0][j], 1, high_conf_idx[1][j], high_conf_idx[2][j]
                ]
            )

        batch_conf.append(conf_list)
        batch_x.append(x_list)
        batch_y.append(y_list)

    final_conf = []
    final_xy = []

    # Perform W-A-S on each image of batch
    for i in range(len(batch_conf)):
        cache_conf, cache_xy = weighted_average_suppression(
            batch_conf[i], batch_x[i], batch_y[i], r_sup
        )
        final_conf.append(np.array(cache_conf))
        final_xy.append(np.round(np.array(cache_xy)+1)) # +1 as DS9 counts pixels from 0 not 1


    return final_conf, final_xy

def scores(true_points, pred_points, r_tp):
    """
    Calculates the number of true positives, false positives
    and false negatives for a single image
    """

    # Deals with images with no predictions
    if len(pred_points) == 0:
        return 0, 0, len(true_points)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for points in true_points:
        if np.count_nonzero(distance.cdist([points], pred_points) < r_tp) >= 1:
            true_positives += 1

    if pred_points.shape[0] > true_positives:
        false_positives = pred_points.shape[0] - true_positives
    
    if true_points.shape[0] > true_positives:
        false_negatives = true_points.shape[0] - true_positives

    return true_positives, false_positives, false_negatives


def completeness(TP, FN):
    return TP / (TP + FN)

def purity(TP, FP):
    # Deals with  case no predictions
    if (TP == 0) and (FP == 0):
        return 0.0
    return TP / (TP + FP)

def find_files(expected_number, path, name_type):
    """
    Function to make a list of all files that exist.
    """
    actual_file_numbers = []
    missing_file_numbers = []
    os.chdir(path)

    count = 0
    for i in range(expected_number):
        if os.path.isfile((name_type + "_" + f"{i:05}" + ".fits")):
            count += 1
            actual_file_numbers.append(f"{i:05}")
        else:
            missing_file_numbers.append(f"{i:05}")

    return actual_file_numbers, missing_file_numbers


def open_fits(file_list_numbers, pixel_x, pixel_y, path, name_type):
    """
    Function to open .fits files and select image layer
    """
    data_images = np.zeros((len(file_list_numbers), 1, pixel_y, pixel_x), dtype = np.float32)

    # moving to right directory
    os.chdir(path)

    print("OPENING FITS...")

    # Opening an selecting appropriate .fits layer
    for count, i in enumerate(file_list_numbers):
        with fits.open(name_type + "_" + i + ".fits") as hdul:
            # hdul.info()
            data_images[count] = np.array(hdul[1].data)

    print("SUCCESSFULLY OPENED FITS \n")

    return data_images

def standardise_images(data_images):
    """
    Function to rescale the pixel values of
    an image to be between 0 and 1 by dividing
    by max pixel value
    """

    print("STANDARDISING PIXEL VALUES...")

    # mean = np.mean(data_images)
    # std = np.std(data_images)
    mx = np.max(data_images)
    print(f"max value is: {mx}")
    data_images /= mx

    print("STANDARDISATION COMPLETE \n")

    return data_images


def open_centres(path_csv, file_list):
    """
    Function to open centres .csv files for each image
    """
    # Note ordering only works up to 99999 as numbers on file names padded to 00000
    csvdir_path = pathlib.Path(path_csv)
    csv_file_list = sorted([str(path) for path in csvdir_path.glob("*.csv")])

    print("OPENING CENTRES")

    centres = []

    # For each image, read the file with centre locations
    for i in range(len(file_list)):
        df_center = pd.read_csv(csv_file_list[i], sep=",", header=None)
        np_center = df_center.values
        np_center[[0, 1]] = np_center[[1, 0]]  # Swapping from y,x to x,y
        centres.append(np_center)
        
    # print("CENTERS DONE \n")

    return centres

def write_regions(confidence, pixel_xy, regions_filename, fits_filename):
    """
    Function to write regions file for single image
    """

    # # Define the colormap and normalization
    # cmap = plt.cm.get_cmap('viridis')
    # norm = plt.Normalize(vmin=0.6, vmax=1)
    cmap = mcolors.LinearSegmentedColormap.from_list('confidence', ['lime', 'lime'])
    norm = mcolors.PowerNorm(gamma=0.0, vmin=0.6, vmax=1.0)


    # Create a list of circle regions at the galaxy positions
    regions_list = []
    for pos, conf in zip(pixel_xy, confidence):
        x, y = pos
        colour = cmap(norm(conf))
        # Convert RGBA color to hexadecimal string
        hex_color = mcolors.to_hex(colour)
        visuals = RegionVisual({'color': hex_color})
        circle_region = regions.CirclePixelRegion(center=regions.PixCoord(x, y),
                                                radius=8,
                                                visual=visuals,
                                                ) # Remove visuals to stop colour

    


        regions_list.append(circle_region)

    with open(regions_filename, 'w') as f:
        f.write('# Region file format: DS9 version 4.0\n')
        for region in regions_list:
            # print((region.serialize(format='ds9')).rstrip() + ' #width = 10 \n' )

            # f.write(region.serialize(format='ds9') + '\n')
            f.write((region.serialize(format='ds9')).rstrip() + ' #width = 3 \n' )

    print(f'Regions file successfully written to: {regions_filename} \n')

    # # Add the regions to your fits file TODO 
    # with fits.open(fits_filename, mode='update') as hdul:
    #     hdul[0].header['REGION'] = (regions_filename)

    print(f'Successfully added regions filename to header of: {fits_filename} \n')



def write_output_csv(confidence, pixel_xy, filename):
    """
    Writes file of confidence, x, y outputs
    """
    header = ['Confidence', 'X', 'Y']
    rows = np.column_stack((confidence,pixel_xy.astype(int)))
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f'Output csv successfully written to: {filename} \n')

def write_metrics_csv(config, filename, r_list, purity_, p_err_, completeness_ , c_err_):
    """
    Writes the calculated metric sweep to a csv file
    """
    with open(config['main_path']+"/"+filename+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(r_list)
        writer.writerow(purity_)
        writer.writerow(p_err_)
        writer.writerow(completeness_)
        writer.writerow(c_err_)

def write_snr_csv(config, filename, snr_list, purity_, purity_err, completeness_ , completeness_err):
    """
    Writes the calculated metric sweep to a csv file
    """
    with open(config['main_path']+"/"+filename+".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(snr_list)
        writer.writerow(purity_)
        writer.writerow(purity_err)
        writer.writerow(completeness_)
        writer.writerow(completeness_err)

def metric_loop(config, fits_names, csv_names, r_tp):
    """
    Function to run a metric loop for a gievn set of parameters:
    Opens fits files ono by one (to save memory) and calculates
    the purity and completeness for each image
    """
    purity_list = []
    completeness_list = []

    ort_session = onnxruntime.InferenceSession(config["onnx_path"])
    input_name = ort_session.get_inputs()[0].name

    for i, name in tqdm(enumerate(fits_names)):
        # Open and standardise image
        with fits.open(name) as hdul:
            test_image = np.array(hdul[1].data).astype(np.float32)
        test_image = np.expand_dims(test_image, 0)
        test_image = np.expand_dims(test_image, 0) / 242.0 # Normalising dataset
        ort_inputs = {input_name: test_image}

        # returns a list: first element is the confidence logits (pre-sigmoid), second is regression values
        ort_outs = ort_session.run(None, ort_inputs)

        # Converts the prediction laebls to real x/y
        confidence, pixel_xy = infer(
            config, ort_outs[0], ort_outs[1], get_anchors(config)
        )

        # getting true centres
        df_center = pd.read_csv(csv_names[i], sep=",", header=None)
        np_center = df_center.values
        np_center[[0, 1]] = np_center[[1, 0]]


        # calculating metrics
        tp, fp, fn = scores(np_center.T, pixel_xy[0], r_tp)

        purity_list.append(purity(tp, fp))
        completeness_list.append(completeness(tp,fn))

    return purity_list, completeness_list