import numpy as np
import torch

x = np.array([[0.2476, 0.2308, 0.2559, 0.3873, 0.4927, 0.4505, 0.3826], 
[0.2548, 0.1447, 0.3391, 0.4523, 0.4956, 0.4330, 0.3837], 
[0.2748, 0.1686, 0.3298, 0.3905, 0.4604, 0.3456, 0.3709], 
[0.3027, 0.1944, 0.3274, 0.3223, 0.3309, 0.3037, 0.3433], 
[0.3206, 0.2861, 0.2797, 0.3904, 0.4078, 0.3879, 0.3486], 
[0.3258, 0.3120, 0.2885, 0.2929, 0.3168, 0.3261, 0.3045]], dtype = np.float64)

mean = np.mean(x,dtype=np.float64)
std = np.std(x,dtype=np.float64)

y = (x-np.full(shape = x.shape, fill_value = mean))

y /= std

print(torch.sigmoid(torch.tensor(y)))