from numpy.core.fromnumeric import reshape
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import ConvTranspose2d


def module():
    return torch.nn.Sequential(
        nn.Conv2d(8, 64, 3, padding=2),
        nn.ReLU(),
        nn.Conv2d(64, 32, 3, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 16, 3, padding=2),
        nn.ReLU(),
        nn.Conv2d(16, 8, 3, padding=2),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 16, 3, padding=2),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 32, 3, padding=2),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 64, 3, padding=2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 8, 3, padding=2)
    )


def penalty():
    pass


def solve(inp):
    pass


def train(inp):
    pass


if __name__ == "__main__":
    pass
