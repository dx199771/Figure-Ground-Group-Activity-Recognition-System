import torch.nn as nn
import torch

import Config
import torch.nn.functional as F


class FigureCNN(nn.Module):
    def __init__(self, num_classes, use_pivot_distances, two_stream):
        super(FigureCNN, self).__init__()
        self.num_joints = 25
        self.use_pivot_distances = use_pivot_distances
        self.device = Config.device
        self.concatenated_dim = 96 if self.use_pivot_distances else 64
        self.num_action_classes = num_classes
        self.num_group_classes = 6
        self.input_size = 2048
        self.num_actors = 8
        self.two_stream = two_stream
        # skeleton data
        self.conv1 = nn.Conv2d(2, 64, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv3 = nn.Conv2d(in_channels=self.num_joints, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                   nn.MaxPool2d(2)
                                   )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(  # feature extracted layer
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        # flatten part ground activity features
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, self.num_group_classes)

    def forward(self, X):

        # Tensor of dim: batch_size, 1024
        group_features = torch.zeros([X.size()[0], 1024], dtype=torch.float).to(Config.device)
        x = X.view(-1, 2, 25, self.num_actors)
        x = x.float()
        # position
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = self.conv3(out)
        out_p = self.conv4(out)
        out = self.conv5(out_p)
        if self.two_stream:
            flatten = torch.flatten(out,start_dim=1)
            return flatten

        # aggregating features at group level
        person_feas = out.view(out.size(0), -1)

        out = self.fc1(person_feas)
        group_output = self.fc2(out)

        assert not ((group_output != group_output).any())  # find out nan in tensor
        return group_output