import logging
import pathlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from lightning.pytorch.accelerators import find_usable_cuda_devices
from pytorch_lightning.loggers import WandbLogger

from .dataset import Dataset
from .model import TotalModel
import numpy as np
import sys
import torch
import h5py
from tqdm import tqdm
from torch import Tensor
import onnx

def my_except_hook(exctype, value, tb_obj):
    """
    For debugging purposes: Prints out the shapes 
    of current tensors if theres an error.
    """
    sys.__excepthook__(exctype, value, tb_obj)
    while tb_obj:
        frame = tb_obj.tb_frame
        for name, local in frame.f_locals.items():
            if type(local) is torch.Tensor:
                print(
                    name,
                    "size: ",
                    local.size(),
                    "type: ",
                    local.dtype,
                    "device: ",
                    local.device,
                )
            elif name == "self":
                for name, local in local.__dict__.items():
                    if type(local) is torch.Tensor:
                        print(
                            "(self)",
                            name,
                            "size: ",
                            local.size(),
                            "type: ",
                            local.dtype,
                            "device: ",
                            local.device,
                        )
        tb_obj = tb_obj.tb_next
sys.excepthook = my_except_hook

def get_torch_anchors(config) -> Tensor:
    """
    For inference
    """
    img_size = config["image_size"]
    feature_size = config["feature_size"]
    step = img_size / feature_size
    halfstep = step * 0.5

    output = np.full((3, 2, feature_size, feature_size), -1, dtype=np.float32)

    # Calculates the lattice of points
    x = np.arange(halfstep, img_size, step, dtype=np.float32)
    y = np.arange(halfstep, img_size, step, dtype=np.float32)
    grid = np.transpose([np.tile(x, len(y)), y.repeat(len(x))]).reshape(
        feature_size, feature_size, 2
    )

    # Could look at doing this quicker but only running once so not too bad.
    # Also because my tensors have so many axis now its difficult to just
    # manipulate with standard .tensor methods.
    for i in range(len(x)):
        for j in range(len(y)):
            for d in range(3):
                output[d][0][i][j] = grid[i][j][0]
                output[d][1][i][j] = grid[i][j][1]

    return torch.tensor(output)

def trial_inference(out_conf, out_reg, torch_anchors, config):
    """
    Performs a basic inference from the model i.e. converts
    the anchor confidence and regression outputs into rough
    x,y pixel coords to test a model in training.
    """
    step = config["image_size"] / config["feature_size"]
    sigmoid = torch.nn.Sigmoid()
    pred_coords = torch.mul(out_reg, step) + torch_anchors
    pred_conf = sigmoid(out_conf)

    batch_conf = []
    batch_x = []
    batch_y = []

    for i in tqdm(range(out_conf.shape[0])):
        high_conf_idx = torch.where(out_conf[i] > config["c_nms"])

        conf_list = []
        x_list = []
        y_list = []

        for j in range(high_conf_idx[0].shape[0]):
            conf_list.append(
                pred_conf[
                    i, high_conf_idx[0][j], high_conf_idx[1][j], high_conf_idx[2][j]
                ]
            )
            x_list.append(
                pred_coords[
                    i, high_conf_idx[0][j], 0, high_conf_idx[1][j], high_conf_idx[2][j]
                ]
            )
            y_list.append(
                pred_coords[
                    i, high_conf_idx[0][j], 1, high_conf_idx[1][j], high_conf_idx[2][j]
                ]
            )

        batch_conf.append(conf_list)
        batch_x.append(x_list)
        batch_y.append(y_list)

    # W-A-S isn't used here as this is just to get a rough gauge of how well
    # a model worked throughout trainin. I.e. is not giving full inferencing
    # because applying the full version would slow model down. 

    return batch_conf[0], batch_x[0], batch_y[0]


logger = logging.getLogger(__name__)
project_root_dir = pathlib.Path(__file__).resolve().parent.parent.parent


if __name__ == "__main__":
    config = {
        "data_path": "/home/ejr85/rds/hpc-work/StaryNight/StaryNight/ppn_stars",
        "data_name": "train_dataset28.hdf5",
        "onnx_path": "/home/ejr85/rds/hpc-work/StaryNight/StaryNight/ONNX",
        "image_size": 224,
        "feature_size": 28,

        # Won't affect training, just used here to guage how training was going
        "c_nms": 0.6, 
        "r_nms": 38,
        "r_suppression": 7,
        "r_tp": 5,
        "r_far": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),
        "r_near": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),

        # Defining traing device
        "accelerator": "cuda",
        "devices": 1,  # find_usable_cuda_devices(1),
        "num_workers": 32,

        # Hyperparameters
        "feature_extractor": "resnet152", #'efficientnet_b0',  #'hrnet_w64',
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 100,
        "N_conf": 1/128.0,
        "N_reg": 1/128.0,
        "pretrained": True,
        "Optimiser": "ADAM",
        "Reg_loss": "Huber",
        "Reg_reduction": "sum",
        "Conf_loss_fn": "CrossEntropy", #"FocalLoss",
        "Conf_reduction": "sum", 
    }

    dataset = Dataset(config)
    model = TotalModel(config)

    # Saves model peridoically
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="/home/ejr85/rds/hpc-work/StaryNight/StaryNight/wandb/checkpoints/",  
        #'/Users/edroberts/Desktop/im_gen/StaryNight/data/checkpoints',
        monitor="val/loss",
    )

    # Starts Logger
    wandb_logger = WandbLogger(log_model="all")

    # log gradients and model topology
    wandb_logger.watch(model)  # , log_freq=3)

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        num_nodes=1,
        devices=config["devices"],
        accelerator=config["accelerator"],
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
    )

    # Running training
    trainer.fit(model, dataset)
    print("Train complete")

    # Saving model to onnx format so that it can easily be transfered around between CUDA and non-CUDA devices
    input_sample = torch.randn((1, 1, config["image_size"], config["image_size"]), device = config["accelerator"] )
    model.to_onnx(config["onnx_path"]+"/model_lowsnr.onnx",
                   input_sample, 
                   export_params=True,
                   input_names = ['input'],  
                   output_names = ['output'], 
                   dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}},
                   verbose=True)

    # Saves model state dictionary
    # torch.save(model.state_dict(),"/home/ejr85/rds/hpc-work/StaryNight/StaryNight/models/test_state_dict")
    # print("Sate dict save complete")


    ##############################################################################
    ################ GETTING ONE PREDICTION - for testing purposes ###############
    ##############################################################################

    run_device = torch.device(config["accelerator"])
    torch_anchors = get_torch_anchors(config).to(run_device)
    points = torch_anchors.unsqueeze(0).expand(config["batch_size"], -1, -1, -1, -1)

    # Send model to GPU (encase read new from checkpoint) and enable evaulation mode
    model.cuda()
    model.eval()

    a = (3, 6, 9)
    for i in a:
        print(f"Image number from 0: {i}")
        # "/home/ejr85/rds/hpc-work/ppn-stary-night/data/dataset.hdf5"
        with h5py.File(config["data_path"] + "/" + config["data_name"], "r") as file:
            item_image = file["images"][i]

        item_image = np.expand_dims(item_image, 0)

        run_device = torch.device(config["accelerator"])
        item_image_torch = torch.tensor(item_image, dtype=torch.float32).to(run_device)
        # predict with the model
        out_conf, out_reg = model(item_image_torch)

        print(torch.sigmoid(out_conf))
        print(out_reg)
        print(trial_inference(out_conf, out_reg, torch_anchors, config))
