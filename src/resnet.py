import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        c = config.NUM_FILTERS
        self.module_list = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        net = self.module_list(x)
        return F.relu(x + net)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        c = config.NUM_FILTERS
        chessboard_size = config.CHESSBOARD_SIZE
        hidden_units = config.VALUE_HEAD_HIDDEN_UNITS

        self.module_list = nn.Sequential(
            nn.Conv2d(2, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            *[ResidualBlock() for _ in range(config.NUM_RESIDUAL_BLOCKS)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(chessboard_size**2 * 2, chessboard_size**2),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(chessboard_size**2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        net = self.module_list(x)
        ret0 = self.policy_head(net).view(
            -1, config.CHESSBOARD_SIZE, config.CHESSBOARD_SIZE
        )
        ret1 = self.value_head(net)[:, 0]
        return (ret0, ret1)


def load_ckpt(path: str, device_id: str) -> ResNet:
    ckpt = torch.load(path, map_location=device_id, weights_only=True)
    network = ResNet()
    network.load_state_dict(ckpt)
    network.to(device_id)
    return network
