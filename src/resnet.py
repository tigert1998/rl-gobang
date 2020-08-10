import torch.nn as nn
import torch.nn.functional as F
from constants import CHESSBOARD_SIZE


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.module_list = nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        net = self.module_list(x)
        return F.relu(x + net)


class ResNet(nn.Module):
    NUM_RESIDUAL_BLOCKS = 19

    def __init__(self):
        super(ResNet, self).__init__()
        self.module_list = nn.Sequential(
            nn.Conv2d(2, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            *[ResidualBlock() for _ in range(self.NUM_RESIDUAL_BLOCKS)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(CHESSBOARD_SIZE ** 2 * 2, CHESSBOARD_SIZE ** 2)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(CHESSBOARD_SIZE ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        net = self.module_list(x)
        return self.policy_head(net), self.value_head(net)
