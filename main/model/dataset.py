import os
import h5py

import torch as torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl


# Class to represent dataset (original PyTorch class)
class gal_dsHDF5(Dataset):
    """
    Dataset class to fetch image data and anchor labels
    for each element of a batch.

    This Dataset uses a HDF5 file to access data instead of
    loading all into memory at once.

    __getitem__
        returns a torch tensor of the index's image,
        confidence labels and regression labels
    """

    def __init__(self, dir_path, filename):
        super().__init__()
        self.dir_path = dir_path
        self.filename = filename

    def __len__(self):
        os.chdir(self.dir_path)
        with h5py.File(self.filename, "r") as hd5file:
            dataset_shape = hd5file["confidence"].shape
        return dataset_shape[0]

    def __getitem__(self, idx):
        # reading preprocessed anchor labels
        os.chdir(self.dir_path)
        with h5py.File(self.filename, "r") as hd5file:
            item_image = hd5file["images"][idx]
            item_conf_gt = hd5file["confidence"][idx]
            item_reg_gt = hd5file["regression"][idx]

        return {
            "images": torch.tensor(item_image, dtype=torch.float32),
            "reg": torch.tensor(item_reg_gt, dtype=torch.float32),
            "conf": torch.tensor(item_conf_gt, dtype=torch.float32),
        }


# Dataset class for pytorch lightning
class Dataset(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.setup()

    def setup(self, stage=None):
        dataset = gal_dsHDF5(self.config["data_path"], self.config["data_name"])
        total = len(dataset)
        train_val = round(total * 0.8)
        lengths = [train_val, total - train_val]
        self.train_dataset, self.val_dataset = torch.utils.data.dataset.random_split(
            dataset, lengths
        )

    def train_dataloader(self, shuffle=None):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=self.config["num_workers"],
        )

    def val_dataloader(self):
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                drop_last=False,
                num_workers=self.config["num_workers"],
            )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=self.config["num_workers"],
        )

    def prod_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()
