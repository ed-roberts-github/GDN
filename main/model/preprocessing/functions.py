from astropy.io import fits
import numpy as np
import os
import pathlib
import pandas as pd
from tqdm import tqdm
import h5py


def find_files(expected_number, path, name_type):
    """
    Function to make a list of all files that exist.

    Gets around missing data in dataset instead of painfully
    going through and regenerating.

    The bug in the generation python script appears to be when
    the max/min number of galaxies are reached.

    It is quicker to simply ignore missing data than sorting
    out the perculiar bug. The total number of images doesn't
    necessarily need to be any number, however should be split in
    batches which are a power of 2.
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
    data_images = np.zeros((len(file_list_numbers), 1, pixel_y, pixel_x))

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


def standardise_images(data_images, pixel_y, pixel_x):
    """
    Function to rescale the pixel values of
    an image to be between 0 and 1 by dividing
    by max pixel value
    """

    img_std = np.zeros((len(data_images), 1, pixel_y, pixel_x))

    print("STANDARDISING PIXEL VALUES...")

    # mean = np.mean(data_images)
    # std = np.std(data_images)
    mx = np.max(data_images)
    print(f"max value is: {mx}")
    data_images /= mx

    print("STANDARDISATION COMPLETE \n")

    return data_images


def get_anchors(config):
    """
    Generates a lattice of anchors for a given image input
    size and feature map size. Note the coordinates are
    (x,y) indexed!

    Partly adapted from Duncan Tilley's work on PPN
    https://github.com/tilleyd/point-proposal-net/

    Returns the coorinates of the origin anchors (x,y).
    """

    # Reads sizes
    img_size = config["image_size"]
    feature_size = config["feature_size"]
    step = img_size / feature_size

    # Calculates step length for given sizes
    halfstep = step * 0.5

    # Calculates the lattice of points
    x = np.arange(halfstep, img_size, step, dtype=np.float32)
    y = np.arange(halfstep, img_size, step, dtype=np.float32)

    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


def get_anchor_labels(anchors, truth_coords, config):
    """
    Creates anchor labels to be used in training. Adds
    the three closest points to each anchor.

    Partly adopted from Duncan Tilley's work on PPN
    https://github.com/tilleyd/point-proposal-net/
    """

    r_near = config["r_near"]
    img_size = config["image_size"]
    feature_size = config["feature_size"]

    step = img_size / feature_size

    # Initialising output
    y_conf = np.full((3, anchors.shape[0]), 0, dtype=np.int8)
    y_reg = np.zeros((3, 2, anchors.shape[0]))

    # For each anchor calculate the distance to each point
    for i in range(len(anchors)):
        x, y = anchors[i]
        x /= step
        y /= step
        distances = []
        indices = []
        for j, (px, py) in enumerate(truth_coords):
            px /= step
            py /= step
            distances.append(np.sqrt((x - px) ** 2 + (y - py) ** 2))
            indices.append(j)

        np_distances = np.array(distances)

        # Adds closest three anchors to list
        if len(distances) > 0:
            ordered_args = np_distances.argsort()

            for centre, arg in enumerate(ordered_args[:3]):
                if distances[indices[arg]] <= r_near:
                    y_conf[centre][i] = 1
                    px, py = truth_coords[indices[arg]]
                    px /= step
                    py /= step

                    y_reg[centre][0][i] = px - x
                    y_reg[centre][1][i] = py - y

    # reshape for use in PPN training
    y_conf = np.reshape(y_conf, (3, feature_size, feature_size))
    y_reg = np.reshape(y_reg, (3, 2, feature_size, feature_size))
    return y_conf, y_reg


def open_and_label_centres(
    path_main, path_csv, conf_labels, reg_labels, file_list, config
):
    """
    Function to open centres .csv files for each image and then
    pass centre coordinates to get_anchor_labels so that we get
    ground truth labesl for each image.

    Also calculates and saves anchors positions.
    """
    # Note ordering only works up to 99999 as numbers on file names padded to 00000
    csvdir_path = pathlib.Path(path_csv)
    csv_file_list = sorted([str(path) for path in csvdir_path.glob("*.csv")])

    # Define origin anchor positions and save to "anchors.npy" file
    anchors = get_anchors(config)
    os.chdir(path_main)
    np.save("anchors", anchors)

    print("CALCULATING LABELS... ")

    # For each image, read the file with centre locations
    for i in tqdm(range(len(file_list))):
        df_center = pd.read_csv(csv_file_list[i], sep=",", header=None)
        np_center = df_center.values
        np_center[[0, 1]] = np_center[[1, 0]]  # Swapping from y,x to x,y

        # Remember .T for np_centers file to transpose from columns to rows
        conf_labels[i], reg_labels[i] = get_anchor_labels(
            anchors, np_center.T, config)

    print("LABELS DONE \n")

    return anchors, conf_labels, reg_labels


def write_HDF5(path, name, image_array, conf_labels, reg_labels):
    """
    Function to write the dataset to HDF5 format
    """
    if len(image_array) != len(conf_labels):
        return "ERROR length image_array != conf_labels"

    if len(image_array) != len(reg_labels):
        return "ERROR length image_array != reg_labels"

    if len(conf_labels) != len(reg_labels):
        return "ERROR length conf_array != reg_labels"

    print("WRITING HDF5 FILE...")

    os.chdir(path)
    with h5py.File(name + ".hdf5", "w") as f:
        f.create_dataset("images", data=image_array)
        f.create_dataset("confidence", data=conf_labels)
        f.create_dataset("regression", data=reg_labels)

    print("HDF5 datafile written")
    return
