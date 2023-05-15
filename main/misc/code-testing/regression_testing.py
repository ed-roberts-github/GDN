# import torch
# from torch import Tensor
# import numpy as np
# import h5py
# import torch.nn as nn

# onfig = {
#     "data_path": "/home/ejr85/rds/hpc-work/StaryNight/StaryNight/ppn_stars",
#     #'/home/ejr85/rds/hpc-work/ppn-stary-night/data',
#     #'/Users/edroberts/Desktop/im_gen/training_data/train/simple',
#     "data_name": "simple_dataset.hdf5",
#     "image_size": 224,
#     "feature_size": 7,
#     "c_nms": 0.5,
#     "r_nms": 0.3,
#     "r_far": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),
#     "r_near": np.sqrt(0.5 * 0.5 + 0.5 * 0.5),
#     "N_conf": 0.5,
#     "N_reg": 1,
#     "Conf_loss_fn": "CrossEntropy",  # "FocalLoss",
#     "batch_size": 4,
#     "lr": 1e-3,
#     "epochs": 70,
#     "num_workers": 32,
#     "accelerator": "cuda",  # 'mps',
#     "devices": 1,  # find_usable_cuda_devices(1),
#     "feature_extractor": "resnet101",
#     "pretrained": True,
#     "Optimiser": "ADAM",
# }


# def get_torch_anchors(config) -> Tensor:
#     img_size = config["image_size"]
#     feature_size = config["feature_size"]
#     step = img_size / feature_size
#     halfstep = step * 0.5

#     output = np.full((3, 2, feature_size, feature_size), -1, dtype=np.float32)

#     # Calculates the lattice of points
#     x = np.arange(halfstep, img_size, step, dtype=np.float32)
#     y = np.arange(halfstep, img_size, step, dtype=np.float32)
#     grid = np.transpose([np.tile(x, len(y)), y.repeat(len(x))]).reshape(
#         feature_size, feature_size, 2
#     )

#     # Could look at doing this quicker but only running once so not too bad.
#     # Also because my tensors have so many axis now its diffuicult to just
#     # manipulate with standard .tensor methods.
#     for i in range(len(x)):
#         for j in range(len(y)):
#             for d in range(3):
#                 output[d][0][i][j] = grid[i][j][0]
#                 output[d][1][i][j] = grid[i][j][1]

#     return torch.tensor(output)


# torch_anchors = get_torch_anchors(onfig)
# points = torch_anchors.unsqueeze(0).expand(onfig["batch_size"], -1, -1, -1, -1)

# # print(torch_anchors)
# # print(points)

# # x = torch.rand((2,6,7,7))
# # print(x)
# # y = x.reshape(-1, 3, 2, 7, 7)
# # print(y)

# conf = torch.randint(0, 2, (4, 3, 7, 7))
# # print(conf)
# # print(torch.eq(conf.unsqueeze(2).expand(-1,-1,2,-1,-1), 1))

# a = 2
# print(f"Image number from 0: {a}")
# # "/home/ejr85/rds/hpc-work/ppn-stary-night/data/dataset.hdf5"
# with h5py.File(
#     "/Users/edroberts/Desktop/im_gen/training_data/train/train/train_dataset.hdf5", "r"
# ) as file:
#     item_image = file["images"][:a]
#     item_conf = file["confidence"][:a]
#     item_reg = file["regression"][:a]

# item_image = np.expand_dims(item_image, 0)
# treg = torch.tensor(item_reg, dtype=torch.float32)
# timage = torch.tensor(item_image, dtype=torch.float32)
# tconf = torch.tensor(item_conf, dtype=torch.float32)

# def reg_loss( reg_logits, reg_gt, conf_gt):
#         """
#         Calculates regression part of loss.

#         Mask ensures only sources contribute to regression.
#         """
#         regLoss = nn.MSELoss(reduction = "sum")
#         pos_mask = torch.eq(conf_gt.unsqueeze(2).expand(-1, -1, 2, -1, -1), 1)
#         return regLoss(reg_logits[pos_mask], reg_gt[pos_mask])





# conf_mask = torch.ne(treg, 0)
# # print(conf_mask)

# test_reg = torch.clone(treg)
# test_reg[0][0][0][2][5] = 0.0
# test_reg[0][0][0][2][4] = 0.0

# print(treg[0][0][0][2][5])
# print(test_reg)
# print(reg_loss(treg,test_reg, tconf))





# def true_points(conf_label, reg_label, self_step, self_torch_anchors):
#     """ 
#     Finds true points from labelling for one image. [[x],[y]]
#     """
#     mask = torch.eq(conf_label.unsqueeze(1).expand(-1, 2, -1, -1), 1)
#     cache = torch.zeros(size=(2,int(torch.sum(conf_label))))
    
#     filled = 0
#     for i in range(3):
#         num_sources = int(torch.sum(conf_label[i]))
#         sources = torch.mul(reg_label[i][mask[i]] , self_step) + self_torch_anchors[i][mask[i]] 
#         shaped_sources = torch.reshape(sources, shape = (2,num_sources))
#         cache[:,filled:filled+num_sources] = shaped_sources
#         filled += num_sources   

#     return torch.unique(cache, dim = 1)

# def true_positives(true_points, pred_points, r_tp):
#     true_count = 0
#     # for i in range(true_points.shape[1]):

#     return



# # print("true points")
# # print(treg[0])
# # print(tconf[0])

# # print(true_points(tconf[1], treg[1], 32, torch_anchors).shape[1])

