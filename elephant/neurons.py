from .lagrangians import *

from torch import Tensor
from typing import *

import torch
import torch.nn as nn

class Neuron(nn.Module):

    def __init__(self,
                 shape,
                 use_bias: bool = False,
                 bias_dims: Set[int] = {},
                 allow_variable_input: bool = False,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32) -> None:
        
        super().__init__()
        self.shape = shape
        self.bias = None
        self.bias_dims = bias_dims
        self.allow_variable_input = allow_variable_input

        if use_bias:
            bias_shape = list(shape)
            if len(bias_dims) > 0:
                for i, s in enumerate(shape):
                    if i+1 in bias_dims:
                        bias_shape[i] = s
                    else:
                        bias_shape[i] = 1
            self.bias = nn.Parameter(0.02*torch.randn(1, *bias_shape, device=device, dtype=dtype))

    def init_state(self,
                   batch_size: int = 1,
                   std: float = 0.02,
                   value: Optional[Tensor] = None,
                   requires_grad: bool = True,
                   device: torch.device = torch.device('cuda'),
                   dtype: torch.dtype = torch.float32) -> Tensor:
        shape_out = [batch_size, *self.shape]
        return std*torch.randn(*shape_out, requires_grad=requires_grad, device=device, dtype=dtype) if value is None else value

    def activation(self, x: Tensor) -> Tensor:
        pass

    def lagrangian(self, x: Tensor) -> Tensor:
        pass

    def energy(self, x: Tensor, g: Tensor) -> Tensor:

        if self.bias is not None:
            if self.allow_variable_input:
                bias_slice = [slice(0, 1, 1)]
                for i, (xs, bs) in enumerate(zip(x.shape[1:], self.bias.shape[1:])):
                    if len(self.bias_dims) == 0 or i+1 in self.bias_dims:
                        assert xs <= bs
                        bias_slice.append(slice(None, min(xs, bs), 1))
                bias = self.bias[*bias_slice]
            else:
                bias = self.bias
            
            xb = x - bias
        
        else:
            xb = x

        b = x.shape[0]
        return torch.mul(g, xb).view(b, -1).sum(dim=1) - self.lagrangian(xb)
    
class LayerNormNeuron(Neuron):

    def __init__(self, *args, dim: int = -1, eps: float = 1e-7, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.eps = eps

    def activation(self, x: Tensor) -> Tensor:
        mu = torch.mean(x, dim=self.dim, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=self.dim, keepdim=True) + self.eps)
        return (x-mu)/std

    def lagrangian(self, x: Tensor) -> Tensor:
        return lagr_layer_norm(x, dim=self.dim, eps=self.eps)
    
class ReluNeuron(Neuron):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def activation(self, x: Tensor) -> Tensor:
        return torch.relu(x)
    
    def lagrangian(self, x: Tensor) -> Tensor:
        return lagr_relu(x)
    
class SoftmaxNeuron(Neuron):

    def __init__(self, *args, dim: int = -1, beta: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.beta = beta

    def activation(self, x: Tensor) -> Tensor:
        return torch.softmax(self.beta*x, dim=self.dim)
    
    def lagrangian(self, x: Tensor) -> Tensor:
        return lagr_softmax(x, dim=self.dim, beta=self.beta)