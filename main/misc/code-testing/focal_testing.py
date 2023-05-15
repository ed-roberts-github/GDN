import numpy as np
import sys
import torch
import h5py
from tqdm import tqdm
from torchvision.utils import _log_api_usage_once
from torch import Tensor
import torch.nn.functional as F

config = {
        "data_path": #"/home/ejr85/rds/hpc-work/StaryNight/StaryNight/ppn_stars",
        '/Users/edroberts/Desktop/im_gen/training_data/train/train',
        #'/Users/edroberts/Desktop/im_gen/training_data/train/simple',
        "data_name": "train_dataset14.hdf5",
        "image_size": 224,
        "feature_size": 14,
        "c_nms": 0.5,
        "r_nms": 38,
        "r_far": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),
        "r_near": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),
        "N_conf": 0.5,
        "N_reg": 0.5,
        "Conf_loss_fn": "FocalLoss", # "CrossEntropy",
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 40,
        "num_workers": 32,
        "accelerator": "cuda",  # 'mps',
        "devices": 1,  # find_usable_cuda_devices(1),
        "feature_extractor": "resnet152",  #'hrnet_w64',
        "pretrained": True,
        "Optimiser": "ADAM",
        "Reg_reduction": "sum",
        "Reg_loss": "Huber"
    }

def sigmoid_focal_loss( 
        #self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
    ) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        Returns:
            Loss tensor with the reduction option applied.
        """
        # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(sigmoid_focal_loss)
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


with h5py.File(config["data_path"] + "/" + config["data_name"], "r") as file:
    item_conf = file["confidence"][:128]

tconf = torch.tensor(item_conf)
trand = torch.randn((128,3,14,14))

print(tconf.shape)

print(sigmoid_focal_loss(inputs = trand,targets = tconf,
        reduction = "sum"
        )/128.0
      )