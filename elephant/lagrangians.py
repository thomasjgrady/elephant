from torch import Tensor
from typing import *

import torch

def lagr_layer_norm(x: Tensor, dim: int = -1, eps: float = 1e-7) -> Tensor:
    b = x.shape[0]
    d = x.shape[dim]
    y = torch.sqrt(torch.var(x, dim=dim) + eps)
    return (d * y).view(b, -1).sum(dim=1)

def lagr_relu(x: Tensor) -> Tensor:
    b = x.shape[0]
    return (0.5*torch.pow(torch.relu(x), 2)).view(b, -1).sum(dim=1)

def lagr_softmax(x: Tensor, dim: int = -1, beta: float = 1.0) -> Tensor:
    b = x.shape[0]
    return 1/beta*torch.logsumexp(beta*x, dim=dim).view(b, -1).sum(dim=1)