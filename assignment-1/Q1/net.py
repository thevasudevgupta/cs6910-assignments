import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(in_features=64, out_features=512)
        self.fc = nn.Linear(in_features=512, out_features=33)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pool(F.relu(x))

        x = self.conv3(x)
        x = self.pool(F.relu(x))

        x = self.conv4(x)
        x = self.pool(F.relu(x))  

        x = self.conv5(x)
        x = F.relu(x)

        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.fc(x)

        return x
