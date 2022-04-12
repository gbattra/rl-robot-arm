# Greg Attra
# 04.11.22

"""
Wrapper classes / utils for nn's
"""

from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, network: nn.Sequential) -> None:
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.network(x)
