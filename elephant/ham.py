from .neurons import Neuron
from .synapses import Synapse

from torch import Tensor
from typing import *

import torch
import torch.nn as nn

class HAM(nn.Module):

    def __init__(self,
                 neurons: Dict[str, Neuron],
                 synapses: Dict[str, Synapse],
                 connections: Dict[str, List[str]]) -> None:
        
        super().__init__()
        
        self.neurons = nn.ModuleDict(neurons)
        self.synapses = nn.ModuleDict(synapses)
        self.connections = connections

    def init_states(self,
                    batch_size: int = 1,
                    stds: Dict[str, float] = {},
                    values: Dict[str, Tensor] = {},
                    requires_grad: bool = True,
                    device: torch.device = torch.device('cuda'),
                    dtype: torch.dtype = torch.float32) -> Dict[str, Tensor]:
        
        return { name: neuron.init_state(
            batch_size=batch_size,
            std=stds.get(name, 0.02),
            value=values.get(name, None),
            requires_grad=requires_grad,
            device=device,
            dtype=dtype
        ) for name, neuron in self.neurons.items() }

    def activations(self, xs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return { name: neuron.activation(xs[name]) for name, neuron in self.neurons.items() }
    
    def neuron_energies(self, xs: Dict[str, Tensor], gs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return { name: neuron.energy(xs[name], gs[name]) for name, neuron in self.neurons.items() }
    
    def neuron_energy(self, xs: Dict[str, Tensor], gs: Dict[str, Tensor]) -> Tensor:
        return torch.cat([e.unsqueeze(1) for e in self.neuron_energies(xs, gs).values()], dim=1).sum(dim=1)

    def synapse_energies(self, gs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        energies = {}
        for name, synapse in self.synapses.items():
            g_neighbors = [gs[neighbor] for neighbor in self.connections[name]]
            energies[name] = synapse.energy(*g_neighbors)
        return energies
    
    def synapse_energy(self, gs: Dict[str, Tensor]) -> Tensor:
        return torch.cat([e.unsqueeze(1) for e in self.synapse_energies(gs).values()], dim=1).sum(dim=1)
    
    def energy(self, xs: Dict[str, Tensor], gs: Dict[str, Tensor]) -> Tensor:
        return self.neuron_energy(xs, gs) + self.synapse_energy(gs)
    
    def dEdg(self, xs: Dict[str, Tensor], gs: Dict[str, Tensor], create_graph: bool = True, return_energy: bool = False) -> Dict[str, Tensor]:
        order = list(sorted(gs.keys()))
        E = self.energy(xs, gs)
        gs_sorted = [gs[name] for name in order]
        grads = torch.autograd.grad(E, gs_sorted, torch.ones_like(E), create_graph=create_graph)
        grads = { name: grad for name, grad in zip(order, grads) }
        if return_energy:
            out = (grads, E)
        else:
            out = grads
        return out     
    
    def energy_descent_step(self,
                            xs: Dict[str, Tensor],
                            gs: Dict[str, Tensor],
                            alphas: Dict[str, float] = {},
                            grad_mask: Dict[str, Tensor] = {},
                            create_graph: bool = True) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        
        dEdg = self.dEdg(xs, gs, create_graph=create_graph)
        for name, mask in grad_mask:
            dEdg[name] = dEdg[name]*mask

        xs_out = { name: x - alphas.get(name, 1.0)*dEdg[name] for name, x in xs.items() }
        return xs_out, self.activations(xs_out)
    
    def energy_descent(self,
                       xs: Dict[str, Tensor],
                       gs: Dict[str, Tensor],
                       alphas: Dict[str, float] = {},
                       grad_mask: Dict[str, Tensor] = {},
                       max_iter: int = 100,
                       tol: float = 1e-3,
                       create_graph: bool = True) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        for t in range(max_iter):
            xs_next, gs_next = self.energy_descent_step(
                xs,
                gs,
                alphas=alphas,
                grad_mask=grad_mask,
                create_graph=create_graph
            )
            with torch.no_grad():
                residuals = { name: torch.norm((gs_next[name] - gs[name]).view(gs[name].shape[0], -1), dim=1) for name in gs.keys() }
            xs, gs = xs_next, gs_next
            with torch.no_grad():
                if all(torch.all(r <= tol) for r in residuals.values()):
                    break

        return xs, gs