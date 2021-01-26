# python 3

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # add conv2D layer
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # add conv2D layer
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # add dropout
        self.dropout1 = nn.Dropout2d(0.25)

        # add FC layer
        self.fc1 = nn.Linear(9216, 128)
        # add label layer
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # pass data through conv layers with relu non linearity
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        # add max pool layer
        x = F.max_pool2d(x, 2)

        # pass through dropout
        x = self.dropout1(x)

        # flatten input
        x = torch.flatten(x, 1)

        # pass through fc layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output