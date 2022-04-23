# Greg Attra
# 04.11.22

"""
Wrapper classes / utils for nn's
"""

from torch import nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
