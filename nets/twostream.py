import torch.nn as nn
import torch

import Config
import torch.nn.functional as F
# two stream network configuration for two-stream variant
class TwoStream(nn.Module):
    def __init__(self,num_action_classes):
        super(TwoStream, self).__init__()
        self.num_action_classes = num_action_classes


        self.fc3 = nn.Sequential(
            nn.Linear(8192, 4096),  # 4*4 for window=64; 8*8 for window=128
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
        self.fc4 = nn.Linear(4096, 1024)
        self.fc5 = nn.Linear(1024, self.num_action_classes)

    def forward(self, input):

        out = self.fc3(input)
        out = self.fc4(out)
        out = self.fc5(out)

        return out