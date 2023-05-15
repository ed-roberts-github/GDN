import numpy as np
from .functions import (
    find_files,
    open_fits,
    standardise_images,
    open_and_label_centres,
    write_HDF5,
)

test_config = {
    "image_size": 224,
    "feature_size": 28,
    "r_far": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),
    "r_near": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),
    "batch_size": 128,
}

prepros_config = {
    "path_to_fits": "/Users/edroberts/Desktop/GDN/training/fits",
    "path_to_csv": "/Users/edroberts/Desktop/GDN/training/csv",
    "path_to_main": "/Users/edroberts/Desktop/GDN/training",
    "image_name": "train",
    "hdf5_name": "train_dataset28",
    "expected_img_number": 300052,
}

if __name__ == "__main__":
    files, missing_files = find_files(
        prepros_config["expected_img_number"],
        prepros_config["path_to_fits"],
        prepros_config["image_name"],
    )

    print("Total number of images in the dataset is: " + str(len(files)) + "\n")

    data_image = open_fits(
        files,
        test_config["image_size"],
        test_config["image_size"],
        prepros_config["path_to_fits"],
        prepros_config["image_name"],
    )

    std_image = standardise_images(
        data_image, test_config["image_size"], test_config["image_size"]
    )

    conf_labels = np.full(
        (len(files), 3, test_config["feature_size"], test_config["feature_size"]),
        -1,
        dtype=np.float64,
    )

    reg_labels = np.full(
        (len(files), 3, 2, test_config["feature_size"], test_config["feature_size"]),
        -1,
        dtype=np.float64,
    )

    anchors, conf_labels, reg_labels = open_and_label_centres(
        prepros_config["path_to_main"],
        prepros_config["path_to_csv"],
        conf_labels,
        reg_labels,
        files,
        test_config,
    )

    write_HDF5(
        prepros_config["path_to_main"],
        prepros_config["hdf5_name"],
        std_image,
        conf_labels,
        reg_labels,
    )
