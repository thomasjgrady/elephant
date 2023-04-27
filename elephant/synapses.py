from .lagrangians import *

from torch import Tensor
from typing import *

import numpy as np
import torch
import torch.nn as nn

class Synapse(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def alignment(self, *gs: Tensor) -> Tensor:
        pass

    def energy(self, *gs: Tensor) -> Tensor:
        return -self.alignment(*gs)
    
class AttentionSynapse(Synapse):

    def __init__(self,
                 n_embed: int,
                 n_heads: int,
                 n_proj: int,
                 beta: float = 1.0,
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float32) -> None:
        
        super().__init__()

        self.wq = nn.Parameter(0.02*torch.randn(n_heads, n_embed, n_proj, device=device, dtype=dtype))
        self.wk = nn.Parameter(0.02*torch.randn(n_heads, n_embed, n_proj, device=device, dtype=dtype))
        self.beta = beta

    def alignment(self, gq: Tensor, gk: Tensor) -> Tensor:
        wq = self.wq / torch.norm(self.wq, dim=1, keepdim=True)
        wk = self.wk / torch.norm(self.wq, dim=1, keepdim=True)
        q = torch.einsum('bte,hez->bhtz', gq, wq)
        k = torch.einsum('bte,hez->bhtz', gk, wk)
        a = (q @ k.transpose(-1, -2))/np.sqrt(q.shape[-1])
        torch.diagonal(a, dim1=-2, dim2=-1).fill_(float('-inf'))
        return lagr_softmax(a, dim=-1, beta=self.beta)
    
class HopfieldSynapse(Synapse):

    def __init__(self,
                 n_in: int,
                 n_hid: int,
                 lagr: Callable = lagr_softmax,
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float32,
                 **lagr_kwargs) -> None:
        
        super().__init__()
        self.W = nn.Parameter(0.02*torch.randn(n_in, n_hid, device=device, dtype=dtype))
        self.lagr = lagr
        self.lagr_kwargs = lagr_kwargs

    def alignment(self, g: Tensor) -> Tensor:
        h = g @ (self.W / torch.norm(self.W, dim=0, keepdim=True))
        return self.lagr(h, **self.lagr_kwargs)