import torch 
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # New regression proposal layers 
        self.reg_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding="same"),
            nn.Dropout(p=0.8),
        )

        self.reg_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding="same"),
            nn.Dropout(p=0.8),
        )

        self.reg_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=6, kernel_size=1, padding="same"),
            nn.Dropout(p=0.8),
        )


    def forward(self, x):
        # reg = self.reg_layer(x[0])
        reg = self.reg_layer1(x)
        print(reg.shape)
        reg = self.reg_layer2(reg)
        print(reg.shape)
        reg = self.reg_layer3(reg)
        print(reg.shape)
        reg = reg.reshape(-1, 3, 2, 7, 7)
        return reg  
    
model = Model()
x = torch.rand((128,2048,7,7))

print(model(x).shape)