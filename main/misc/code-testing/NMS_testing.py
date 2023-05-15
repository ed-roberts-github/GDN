import torch
import numpy as np
from tqdm import tqdm
from torch import tensor
import h5py
from scipy.spatial.distance import cdist

onfig = {
    "data_path": "/home/ejr85/rds/hpc-work/StaryNight/StaryNight/ppn_stars",
    #'/home/ejr85/rds/hpc-work/ppn-stary-night/data',
    #'/Users/edroberts/Desktop/im_gen/training_data/train/simple',
    "data_name": "simple_dataset.hdf5",
    "image_size": 224,
    "feature_size": 7,
    "c_nms": 0.5,
    "r_nms": 0.3,
    "r_far": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),
    "r_near": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),
    "N_conf": 0.5,
    "N_reg": 1,
    "Conf_loss_fn": "CrossEntropy",  # "FocalLoss",
    "batch_size": 4,
    "lr": 1e-3,
    "epochs": 70,
    "num_workers": 32,
    "accelerator": "cuda",  # 'mps',
    "devices": 1,  # find_usable_cuda_devices(1),
    "feature_extractor": "resnet101",
    "pretrained": True,
    "Optimiser": "ADAM",
}


def get_torch_anchors():
    img_size = 224
    feature_size = 28
    step = img_size / feature_size
    halfstep = step * 0.5

    output = np.full((3, 2, feature_size, feature_size), -1, dtype=np.float32)
    # Calculates the lattice of points
    x = np.arange(halfstep, img_size, step, dtype=np.float32)
    y = np.arange(halfstep, img_size, step, dtype=np.float32)
    grid = np.transpose([np.tile(x, len(y)), y.repeat(len(x))]).reshape(
        feature_size, feature_size, 2
    )

    for i in range(len(x)):
        for j in range(len(y)):
            for d in range(3):
                output[d][0][i][j] = grid[i][j][0]
                output[d][1][i][j] = grid[i][j][1]

    return torch.tensor(output)


def predict_step(anchor_points, r_sup):
    step = 32
    c_nms = 0.7
    batch_size = 2

    # Pass images through forward to get output predictions
    out_conf = torch.randn((batch_size, 3, 7, 7))
    out_reg = torch.randn((batch_size, 3, 2, 7, 7))
    out_pixels = torch.mul(out_reg, step) + anchor_points

    batch_conf = []
    batch_x = []
    batch_y = []

    for i in tqdm(range(out_conf.shape[0])):
        high_conf_idx = torch.where(out_conf[i] > c_nms)
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
        final_conf.append(cache_conf)
        final_xy.append(cache_xy)

    return final_conf, final_xy


def NMS(out_confs, pred_x, pred_y, r_nms):
    """
    Performs Non-maximum suppression to remove duplicate
    points.

    Inputs are intended to be for a single image.
    """
    NMS_conf = []
    NMS_x = []
    NMS_y = []
    order_args = np.ndarray.tolist(np.flip(np.array(out_confs).argsort()))

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
    Removes duplicate predictions by performing
    a weighted average of the points.

    Inputs are intended to be for a single image.
    """
    predicted_xy = np.ndarray.tolist(np.column_stack((pred_x, pred_y)))
    suppressed_conf = []
    suppressed_xy = []

    while predicted_xy != []:
        distances = cdist([predicted_xy[0]], predicted_xy)
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

        else:
            raise Exception(
                "More than one prediction within suppression range: implment higher averaging"
            )

    return suppressed_conf, suppressed_xy

def real_points(conf_label, reg_label, self_step, self_torch_anchors):
    """
    Finds real points of sources of one image from its labelling. [[x,y]]
    """
    mask = torch.eq(conf_label.unsqueeze(1).expand(-1, 2, -1, -1), 1)
    cache = torch.zeros(size=(2, int(torch.sum(conf_label))))
    filled = 0
    for i in range(3):
        num_sources = int(torch.sum(conf_label[i]))
        sources = (
            torch.mul(reg_label[i][mask[i]], self_step) + self_torch_anchors[i][mask[i]]
        )
        shaped_sources = torch.reshape(sources, shape=(2, num_sources))
        cache[:, filled : filled + num_sources] = shaped_sources
        filled += num_sources

    return torch.unique(cache, dim=1).T

# def real_points(self, conf_label, reg_label):
    #     """
    #     Finds real points of sources of one image from its labelling. [[x,y]]
    #     """
    #     mask = torch.eq(conf_label.unsqueeze(1).expand(-1, 2, -1, -1), 1)
    #     cache = torch.zeros(size=(2, int(torch.sum(conf_label))))
    #     filled = 0
    #     for i in range(3):
    #         num_sources = int(torch.sum(conf_label[i]))
    #         sources = (
    #             torch.mul(reg_label[i][mask[i]], self.step)
    #             + self.torch_anchors[i][mask[i]]
    #         )
    #         shaped_sources = torch.reshape(sources, shape=(2, num_sources))
    #         cache[:, filled : filled + num_sources] = shaped_sources
    #         filled += num_sources

    #     return torch.unique(cache, dim=1).T


def scores(true_points, pred_points, r_tp):
    """
    Calculates the number of true positives, false positives
    and false negatives for a single image
    """
    true_positives = 0
    for points in (true_points):
        if np.count_nonzero(cdist([points], pred_points) < r_tp) >= 1:
            true_positives += 1

    false_positives = pred_points.shape[0] - true_positives
    false_negatives = true_points.shape[0] - true_positives

    return true_positives, false_positives, false_negatives

def completeness(TP, FN):
    return TP / (TP + FN)

def purity(TP, FP):
    return TP / (TP + FP)


a = 2
print(f"Image number from 0: {a}")
# "/home/ejr85/rds/hpc-work/ppn-stary-night/data/dataset.hdf5"
with h5py.File("/Users/edroberts/Desktop/im_gen/training_data/train/train/train_dataset28.hdf5", "r") as file:
    item_image = file["images"][:a]
    item_conf = file["confidence"][:a]
    item_reg = file["regression"][:a]

item_image = np.expand_dims(item_image, 0)
treg = torch.tensor(item_reg, dtype=torch.float32)
timage = torch.tensor(item_image, dtype=torch.float32)
tconf = torch.tensor(item_conf, dtype=torch.float32)

torch_anchors = get_torch_anchors()
points = torch_anchors.unsqueeze(0).expand(1,-1,-1,-1,-1)
tp = real_points(tconf[0], treg[0], 8, torch_anchors)

print(tp)



# x = [1, 5, 7]
# y = [1, 5, 6]

# q = [1, 2, 3]
# p = [1, 5, 6]

# bob = np.column_stack((x, y))
# steve = np.column_stack((q, p))

# # print(bob)
# # print(steve)
# # print(cdist(bob,[[0,0]]))
# # print(np.where(cdist(bob,bob)<6))
# # print(cdist([steve[0]],steve))
# x = [[1, 1], [2, 2]]

# print((1 * np.array(x[0]) + 2 * np.array(x[1])) / (1 + 2))


# confs = [ 0.63, 0.93, 0.94, 0.88, 0.93, 0.7, 0.76]
# x = [108.5, 114.8, 100, 188.3, 188.3, 116.7, 123.2]
# y = [ 52, 52, 100, 93.0, 99.0, 27.8, 27.9]

# print((np.array(weighted_average_suppression(confs, x, y, 7)[1])))
