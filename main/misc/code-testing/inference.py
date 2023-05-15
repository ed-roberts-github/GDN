# Script for simple on image predictions for manual testing

# from .model import TotalModel
# import pytorch_lightning as pl
# import torch
# from torch import Tensor

# import h5py
from astropy.io import fits
import os
from tqdm import tqdm
import pathlib 
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
                "Weighted average suppression: More than one prediction within suppression \
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

    for i in tqdm(range(out_conf.shape[0])):
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
    true_positives = 0
    for points in true_points:
        if np.count_nonzero(distance.cdist([points], pred_points) < r_tp) >= 1:
            true_positives += 1

    false_positives = pred_points.shape[0] - true_positives
    false_negatives = true_points.shape[0] - true_positives

    return true_positives, false_positives, false_negatives


def completeness(TP, FN):
    return TP / (TP + FN)


def purity(TP, FP):
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
    for i in tqdm(range(len(file_list))):
        df_center = pd.read_csv(csv_file_list[i], sep=",", header=None)
        np_center = df_center.values
        np_center[[0, 1]] = np_center[[1, 0]]  # Swapping from y,x to x,y
        centres.append(np_center)
        
    print("CENTERS DONE \n")

    return centres






config_infer = {
    "fits_path": "/Users/edroberts/Desktop/im_gen/training_data/train/test/fits/",
    "onnx_path": "/Users/edroberts/Desktop/im_gen/StaryNight/onnx/",

    "image_size": 224,
    "feature_size": 28,
    "c_cutoff": 0.6,
    "r_suppression": 7.5,
    "r_tp": 5,

    "accelerator": "mps",  # "cuda",
    "devices": 1,  # find_usable_cuda_devices(1),
    "num_workers": 2,  # 32,
    "feature_extractor": "resnet152",  #'hrnet_w64',
    "batch_size": 128,
    "lr": 1e-3,
    "epochs": 100,
    "N_conf": 1 / 128.0,
    "N_reg": 1 / 128.0,
    "pretrained": True,
    "Optimiser": "ADAM",
    "Reg_loss": "Huber",
    "Reg_reduction": "sum",
    "Conf_loss_fn": "CrossEntropy",  # "FocalLoss",
    "Conf_reduction": "sum",
}

################# To get preidtions for an unlabelled image #####################

ort_session = onnxruntime.InferenceSession(config_infer["onnx_path"] + "model-195.onnx")
input_name = ort_session.get_inputs()[0].name

with fits.open(config_infer["fits_path"] + "testing_00002.fits") as hdul:
    test_image = np.array(hdul[1].data).astype(np.float32)
test_image = np.expand_dims(test_image, 0)

# Standardising, need to do this another way eventually.
test_image = np.expand_dims(test_image, 0) / 242.06 

ort_inputs = {input_name: test_image}

# returns a list: first element is the confidence logits (pre-sigmoid), second is regression values
ort_outs = ort_session.run(None, ort_inputs)

# Converts the prediction laebls to real x/y
confidence, pixel_xy = infer(
    config_infer, ort_outs[0], ort_outs[1], get_anchors(config_infer)
)

print(f"Confidence values: \n {confidence} \n")
print(f"DS9 pixel (x,y): \n {pixel_xy} \n")

#####################################################################################


################# To get purity and completeness ####################################
#

##################################################################################
#   This is old code used for referecning via other methods used during
#   my training experiments. I have kept it encase it is required again.
#                               \/\/\/
########################### checkpointing ########################################
# run_device = torch.device(config["accelerator"])
# torch_anchors = get_torch_anchors(config).to(run_device)
# points = torch_anchors.unsqueeze(0).expand(1,-1,-1,-1,-1) # 1 = batch_size as only one image

# #"/Users/edroberts/Desktop/im_gen/StaryNight/wandb/checkpoints/epoch=98-step=18414.ckpt"
# Model = TotalModel.load_from_checkpoint("/home/ejr85/rds/hpc-work/StaryNight/StaryNight/wandb/checkpoints/epoch=98-step=18216-v1.ckpt")

# Model.cuda()# put model on device
# Model.eval()# disable randomness, dropout, etc...

# with fits.open("/home/ejr85/rds/hpc-work/data/train/fits/train_46220.fits") as hdul:
#         # hdul.info()
#         test_image = np.array(hdul[1].data).astype(np.float32)

# test_image = np.expand_dims(test_image,0)
# test_image = np.expand_dims(test_image,0)/242.06
# print(np.shape(test_image))
# test_image_torch = torch.tensor(test_image, dtype=torch.float32).to(run_device)

# # print(test_image_torch.shape)
# # predict with the model
# with torch.no_grad():
#     out_conf, out_reg = Model(test_image_torch)

# print("test image train_46219.fits")
# print(torch.sigmoid(out_conf))
# print(out_reg)

# c,x,y = infer(config, torch.sigmoid(out_conf), out_reg, torch_anchors)

# print(c)
# print(x)
# print(y)
########################### checkpointing ########################################
#
# print("----------------------------------------------------------------------")
#
# a=(3,6,9)
# for i in a:
#     # "/Users/edroberts/Desktop/im_gen/StaryNight/datadataset.hdf5"
#     with h5py.File("/home/ejr85/rds/hpc-work/StaryNight/StaryNight/ppn_stars/simple_dataset.hdf5", "r") as file:
#         item_image = file['images'][i]
#
#     item_image = np.expand_dims(item_image,0)
#     print(item_image.shape)
#
#     item_image_torch = torch.tensor(item_image, dtype=torch.float32).to(run_device)
#     # predict with the model
#     out_conf, out_reg = Model(item_image_torch)
#
#
#     print(torch.sigmoid(out_conf))
#     print(out_reg)
#     print(infer(out_conf, out_reg, torch_anchors, config))
