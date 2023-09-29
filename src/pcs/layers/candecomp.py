from typing import List, Tuple

import torch
from torch import nn

from pcs.initializers import init_params_
from pcs.layers import MonotonicComputeLayer, BornComputeLayer
from pcs.utils import safelog
from region_graph import RegionNode


class MonotonicCPLayer(MonotonicComputeLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_in_components: int,
            num_out_components: int,
            init_method: str = 'dirichlet',
            init_scale: float = 1.0
    ):
        super().__init__(rg_nodes, num_in_components, num_out_components)
        weight = torch.empty(len(rg_nodes), num_out_components, num_in_components)
        init_params_(weight, init_method, init_scale=init_scale)
        self.weight = nn.Parameter(torch.log(weight), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_folds, arity, num_in_components)
        # Compute the element-wise product
        x = torch.sum(x, dim=-2)  # (-1, num_folds, num_in_components)

        # Log-einsum-exp trick
        w = torch.exp(self.weight)
        m_x, _ = torch.max(x, dim=-1, keepdim=True)  # (-1, num_folds, 1)
        e_x = torch.exp(x - m_x)                     # (-1, num_folds, num_in_components)
        y = torch.einsum('fkp,bfp->bfk', w, e_x)     # (-1, num_folds, num_out_components)
        y = m_x + safelog(y)                       # (-1, num_folds, num_out_components)
        return y


class BornCPLayer(BornComputeLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_in_components: int,
            num_out_components: int,
            init_method: str = 'normal',
            init_scale: float = 1.0,
            exp_reparam: bool = False
    ):
        super().__init__(rg_nodes, num_in_components, num_out_components)
        weight = torch.empty(len(rg_nodes), num_out_components, num_in_components)
        init_params_(weight, init_method, init_scale=init_scale)
        if exp_reparam:
            weight = torch.log(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.exp_reparam = exp_reparam

    def forward(self, x: torch.Tensor, x_si: torch.Tensor, square: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = torch.exp(self.weight) if self.exp_reparam else self.weight

        if square:
            # x: (-1, num_folds, num_in_components, num_in_components)
            # Compute the element-wise product
            x = torch.sum(x, dim=-3)         # (-1, num_folds, num_in_components, num_in_components)
            x_si = torch.prod(x_si, dim=-3)  # (-1, num_folds, num_in_components, num_in_components)

            # Non-monotonic Log-einsum-exp trick
            m_x, _ = torch.max(x, dim=-2, keepdim=True)  # (-1, num_folds, 1, num_in_components)
            e_x = x_si * torch.exp(x - m_x)              # (-1, num_folds, num_in_components, num_in_components)
            # x: (-1, num_folds, num_out_components, num_in_components)
            x = torch.einsum('fki,bfij->bfkj', weight, e_x)
            x_si = torch.sign(x.detach())
            x = m_x + safelog(torch.abs(x))
            m_x, _ = torch.max(x, dim=-1, keepdim=True)  # (-1, num_folds, num_out_components, 1)
            e_x = x_si * torch.exp(x - m_x)              # (-1, num_folds, num_out_components, num_in_components)
            # (-1, num_folds, num_out_components, num_out_components)
            x = torch.einsum('bfkj,flj->bfkl', e_x, weight)
            x_si = torch.sign(x.detach())
            x = m_x + safelog(torch.abs(x))
            return x, x_si

        # x: (-1, num_folds, arity, num_in_components)
        # Compute the element-wise product
        x = torch.sum(x, dim=-2)         # (-1, num_folds, num_in_components)
        x_si = torch.prod(x_si, dim=-2)  # (-1, num_folds, num_in_components)

        # Non-monotonic Log-einsum-exp trick
        m_x, _ = torch.max(x, dim=-1, keepdim=True)  # (-1, num_folds, 1)
        e_x = x_si * torch.exp(x - m_x)              # (-1, num_folds, num_in_components)
        # y: (-1, num_folds, num_out_components)
        y = torch.einsum('fkp,bfp->bfk', weight, e_x)
        y_si = torch.sign(y.detach())
        y = m_x + safelog(torch.abs(y))
        return y, y_si
