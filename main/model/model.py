import os
import pathlib
import h5py

from tqdm import tqdm
import numpy as np

import torch as torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
from torchvision.utils import _log_api_usage_once

# from torchinfo import summary
import torch.nn as nn
import timm
from torch import Tensor
from .pl_model import PLModel



class TotalModel(PLModel):
    def __init__(self, config):
        super().__init__(config)
        self.feature_size = self.config["feature_size"]
        self.image_size = self.config["image_size"]
        self.step = self.image_size / self.feature_size
        self.batch_key = "images"
        self.run_device = torch.device(self.config["accelerator"])
        self.torch_anchors = self.get_torch_anchors().to(self.run_device)
        self.points = self.torch_anchors.unsqueeze(0).expand(
            self.config["batch_size"], -1, -1, -1, -1
        )

        # outputs [?,512,feature_size,feature_size] -Note: 512 will change depending on backbone
        self.resnet = timm.create_model(
            self.config["feature_extractor"],
            features_only=True,
            out_indices=[2],
            pretrained=self.config["pretrained"],
            in_chans=1,
        )

        # Confidence proposal layer - [?,3,feature_size,feature_size]
        self.conf_layer = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=3, kernel_size=1, padding=0), #"same"),
            nn.Dropout(p=0.8),
        )

        # Regression proposal layer - [?,6,feature_size,feature_size]
        self.reg_layer = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=6, kernel_size=1, padding=0), #"same"),
            nn.Dropout(p=0.8),
        )

    def forward(self, x):
        """
        Describes the path data follows through the network
        """
        x = self.resnet(x)
        conf_logits = self.conf_layer(x[0])
        reg = self.reg_layer(x[0])
        reg = reg.reshape(-1, 3, 2, self.feature_size, self.feature_size)
        return conf_logits, reg

    def _evaluate(self, batch, current_step):
        """
        Tells PL how to evaluate the loss.
        """
        out_conf, out_reg = self(batch["images"])
        return {"loss": self.total_loss(out_reg, batch["reg"], out_conf, batch["conf"])}


    def sigmoid_focal_loss(
        self,
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
            _log_api_usage_once(self.sigmoid_focal_loss)
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

    def reg_loss(self, reg_logits, reg_gt, conf_gt):
        """
        Calculates regression part of loss.
        """
        # Mask ensures only sources contribute to regression
        pos_mask = torch.eq(conf_gt.unsqueeze(2).expand(-1, -1, 2, -1, -1), 1)

        if self.config["Reg_loss"] == "MSE":
            regLoss = nn.MSELoss(reduction=self.config["Reg_reduction"])
            return regLoss(reg_logits[pos_mask], reg_gt[pos_mask])

        elif self.config["Reg_loss"] == "L1":
            regLoss = nn.L1Loss(reduction=self.config["Reg_reduction"])
            return regLoss(reg_logits[pos_mask], reg_gt[pos_mask])

        elif self.config["Reg_loss"] == "Huber":
            regLoss = nn.HuberLoss(reduction=self.config["Reg_reduction"], delta=1.0)
            return regLoss(reg_logits[pos_mask], reg_gt[pos_mask])

        else:
            raise Exception("Regression loss undefined")

    def total_loss(self, reg_logits, reg_gt, conf_logits, conf_gt):
        """
        Calculate the total loss function.
        """

        regression_loss = self.reg_loss(reg_logits, reg_gt, conf_gt)

        conf_mask = torch.ne(conf_gt, -1)

        if self.config["Conf_loss_fn"] == "CrossEntropy":
            BCE = torch.nn.BCEWithLogitsLoss(
                weight=None,
                size_average=None,
                reduce=None,
                reduction=self.config["Conf_reduction"],
                pos_weight=None,
            )

            confidence_loss = BCE(conf_logits[conf_mask], conf_gt[conf_mask])

        elif self.config["Conf_loss_fn"] == "FocalLoss":
            confidence_loss = self.sigmoid_focal_loss(
                inputs=conf_logits[conf_mask],
                targets=conf_gt[conf_mask],
                alpha=0.25,
                gamma=2,
                reduction=self.config["Conf_reduction"],
            )

        else:
            raise Exception("Need to implement other loss")

        N_conf, N_reg = self.config["N_conf"], self.config["N_reg"]

        return N_conf * confidence_loss + N_reg * regression_loss

    def predict_step(self, batch, batch_idx):
        """
        Tells PL how to run a prediction.

        !!! Note: this function is not used here in the end as all proper 
        inferencing was done after the model was saved in onnx format.
        see /inferencing/ !!!

        Also Note: the use of python lists instead of numpy arrays
        or torch tensors is because the number of outputs must
        be allowed to vary so static shape of np and tensors
        means they can't be used effectively.
        """

        # Passes image through forward
        out_conf, out_reg = self(batch["images"])

        # Calculate real (x,y) coords and confidence probability
        sigmoid = nn.Sigmoid()
        pred_coords = (
            torch.mul(out_reg, self.step)
            + self.points 
        )
        pred_conf = sigmoid(out_conf)

        batch_conf = []
        batch_x = []
        batch_y = []

        # Selecting points with confidence score above c_nms
        for i in tqdm(range(out_conf.shape[0])):
            high_conf_idx = torch.where(out_conf[i] > self.config["c_nms"])

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
                        i,
                        high_conf_idx[0][j],
                        0,
                        high_conf_idx[1][j],
                        high_conf_idx[2][j],
                    ]
                )
                y_list.append(
                    pred_coords[
                        i,
                        high_conf_idx[0][j],
                        1,
                        high_conf_idx[1][j],
                        high_conf_idx[2][j],
                    ]
                )

            batch_conf.append(conf_list)
            batch_x.append(x_list)
            batch_y.append(y_list)

        # final_conf = []
        # final_xy = []

        # # Perform W-A-S on each image of batch
        # for i in range(len(batch_conf)):
        #     cache_conf, cache_xy = self.weighted_average_suppression(
        #         batch_conf[i], batch_x[i], batch_y[i]
        #     )
        #     final_conf.append(cache_conf)
        #     final_xy.append(cache_xy)

        # TODO Complete final stage testing of this function
        # return final_conf, final_xy
        return batch_conf, batch_x, batch_y

    def NMS(self, out_confs, pred_x, pred_y):
        """
        Performs Non-maximum suppression to remove duplicate
        points.

        Inputs are intended to be for a single image.

        This is a retired function used before the adoption of 
        WAS. 
        
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
                    (NMS_x[count] - pred_x[arg]) ** 2
                    + (NMS_y[count] - pred_y[arg]) ** 2
                )
                if distance <= self.config["r_nms"]:
                    to_remove.append(arg)

            [order_args.remove(arg) for arg in to_remove]
            count += 1

        return NMS_conf, NMS_x, NMS_y

    # def weighted_average_suppression(self, out_confs, pred_x, pred_y):
    #     """
    #     Removes duplicate predictions by performing
    #     a weighted average of the points.

    #     Inputs are intended to be for a single image.

    #     !!! Note: this function is not used here in the end as all proper 
    #     inferencing was done after the model was saved in onnx format.
    #     see /inferencing !!!
    #     """
    #     predicted_xy = np.ndarray.tolist(np.column_stack((pred_x, pred_y)))
    #     suppressed_conf = []
    #     suppressed_xy = []

    #     while predicted_xy != []:
    #         distances = distance.cdist([predicted_xy[0]], predicted_xy)
    #         indices = [
    #             i
    #             for (i, dist) in enumerate(distances[0])
    #             if dist < self.config["r_suppression"]
    #         ]

    #         if len(indices) == 1:
    #             suppressed_conf.append(out_confs[0])
    #             suppressed_xy.append(predicted_xy[0])

    #             del predicted_xy[0]
    #             del out_confs[0]

    #         elif len(indices) == 2:
    #             ave_xy = (
    #                 np.array(predicted_xy[0]) * out_confs[0]
    #                 + np.array(predicted_xy[indices[1]]) * out_confs[indices[1]]
    #             ) / (out_confs[0] + out_confs[indices[1]])

    #             suppressed_conf.append(np.mean((out_confs[0], out_confs[indices[1]])))
    #             suppressed_xy.append(np.ndarray.tolist(ave_xy))

    #             del predicted_xy[indices[1]]
    #             del out_confs[indices[1]]
    #             del predicted_xy[0]
    #             del out_confs[0]

    #       elif len(indices) == 3:
    #                ave_xy = (
    #                    np.array(predicted_xy[0]) * out_confs[0]
    #                    + np.array(predicted_xy[indices[1]]) * out_confs[indices[1]]
    #                    + np.array(predicted_xy[indices[2]]) * out_confs[indices[2]]
    #                ) / (out_confs[0] + out_confs[indices[1]] + out_confs[indices[2]])

    #                suppressed_conf.append(np.mean((out_confs[0], out_confs[indices[1]])))
    #                suppressed_xy.append(np.ndarray.tolist(ave_xy))

    #                del predicted_xy[indices[2]]
    #                del out_confs[indices[2]]
    #                del predicted_xy[indices[1]]
    #                del out_confs[indices[1]]
    #                del predicted_xy[0]
    #                del out_confs[0]

    #         else:
    #             raise Exception(
    #                 "More than one prediction within suppression range: implment higher averaging"
    #             )

    #     return suppressed_conf, suppressed_xy


    def get_torch_anchors(self) -> Tensor:
        """
        Generates a lattice of anchors for a given image input
        size and feature map size. Coordinates are (x,y) indexed.

        Using loops not vectorisatinon because tensors have too
        many axes (5) to manipulate easily with standard tensor methods.
        """
        step = self.step
        halfstep = step * 0.5

        output = np.full(
            (3, 2, self.feature_size, self.feature_size), -1, dtype=np.float32
        )

        # Calculates the lattice of points
        x = np.arange(halfstep, self.image_size, step, dtype=np.float32)
        y = np.arange(halfstep, self.image_size, step, dtype=np.float32)
        grid = np.transpose([np.tile(x, len(y)), y.repeat(len(x))]).reshape(
            self.feature_size, self.feature_size, 2
        )

        for i in range(len(x)):
            for j in range(len(y)):
                for d in range(3):
                    output[d][0][i][j] = grid[i][j][0]
                    output[d][1][i][j] = grid[i][j][1]

        return torch.tensor(output)
