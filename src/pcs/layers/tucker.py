from typing import List, Tuple

import torch
from torch import nn

from pcs.initializers import init_params_
from pcs.layers import MonotonicComputeLayer, BornComputeLayer
from pcs.utils import safelog
from region_graph import RegionNode


class MonotonicTucker2Layer(MonotonicComputeLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_in_components: int,
            num_out_components: int,
            init_method: str = 'dirichlet',
            init_scale: float = 1.0
    ):
        super().__init__(rg_nodes, num_in_components, num_out_components)
        weight = torch.empty(len(rg_nodes), num_out_components, num_in_components, num_in_components)
        init_params_(weight, init_method, init_scale=init_scale)
        self.weight = nn.Parameter(torch.log(weight), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_folds, arity, num_in_components)
        assert x.shape[2] == 2
        # lx: (-1, num_folds, num_in_components)
        # rx: (-1, num_folds, num_in_components)
        lx, rx = x[:, :, 0], x[:, :, 1]

        # Log-einsum-exp trick
        w = torch.exp(self.weight)
        m_lx, _ = torch.max(lx, dim=-1, keepdim=True)  # (-1, num_folds, 1)
        m_rx, _ = torch.max(rx, dim=-1, keepdim=True)  # (-1, num_folds, 1)
        e_lx = torch.exp(lx - m_lx)                    # (-1, num_folds, num_in_components)
        e_rx = torch.exp(rx - m_rx)                    # (-1, num_folds, num_in_components)
        # y: (-1, num_folds, num_out_components)
        y = torch.einsum('fkpq,bfp,bfq->bfk', w, e_lx, e_rx)
        y = m_lx + m_rx + safelog(y)
        return y


class BornTucker2Layer(BornComputeLayer):
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

        weight = torch.empty(len(rg_nodes), num_out_components, num_in_components, num_in_components)
        init_params_(weight, init_method, init_scale=init_scale)
        if exp_reparam:
            weight = torch.log(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.exp_reparam = exp_reparam

    def forward(self, x: torch.Tensor, x_si: torch.Tensor, square: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = torch.exp(self.weight) if self.exp_reparam else self.weight

        if square:
            # x: (-1, num_folds, 2, num_in_components, num_in_components)
            assert x.shape[2] == 2
            # lx: (-1, num_folds, num_in_components, num_in_components)
            # rx: (-1, num_folds, num_in_components, num_in_components)
            lx, rx = x[:, :, 0], x[:, :, 1]
            lx_si, rx_si = x_si[:, :, 0], x_si[:, :, 1]

            # Non-monotonic Log-einsum-exp trick
            m_lx, _ = torch.max(lx, dim=-2, keepdim=True)  # (-1, num_folds, 1, num_in_components)
            m_rx, _ = torch.max(rx, dim=-1, keepdim=True)  # (-1, num_folds, num_in_components, 1)
            e_lx = lx_si * torch.exp(lx - m_lx)  # (-1, num_folds, num_in_components, num_in_components)
            e_rx = rx_si * torch.exp(rx - m_rx)  # (-1, num_folds, num_in_components, num_in_components)
            #
            # lx: (-1, num_folds, num_out_components, num_in_components, num_in_components)
            lx = torch.einsum('fpij,bfil->bfpjl', weight, e_lx)
            lx_si = torch.sign(lx.detach())
            lx = m_lx.unsqueeze(dim=-3) + safelog(torch.abs(lx))
            # rx: (-1, num_folds, num_out_components, num_in_components, num_in_components)
            rx = torch.einsum('fqlm,bfjm->bfqjl', weight, e_rx)
            rx_si = torch.sign(rx.detach())
            rx = m_rx.unsqueeze(dim=-3) + safelog(torch.abs(rx))
            # x: (-1, num_folds, num_out_components, num_out_components)
            m_lx = torch.amax(lx, dim=(-2, -1), keepdim=True)  # (-1, num_folds, num_out_components, 1, 1)
            m_rx = torch.amax(rx, dim=(-2, -1), keepdim=True)  # (-1, num_folds, num_out_components, 1, 1)
            e_lx = lx_si * torch.exp(lx - m_lx)
            e_rx = rx_si * torch.exp(rx - m_rx)
            # x: (-1, num_folds, num_out_components, num_out_components)
            x = torch.einsum('bfpjl,bfqjl->bfpq', e_lx, e_rx)
            x_si = torch.sign(x.detach())
            x = m_lx.squeeze(dim=-1) + m_rx.squeeze(dim=-1).squeeze(dim=-1).unsqueeze(dim=-2) + safelog(torch.abs(x))
            return x, x_si

        # x: (-1, num_folds, arity, num_in_components)
        assert x.shape[2] == 2
        # lx: (-1, num_folds, num_in_components)
        # rx: (-1, num_folds, num_in_components)
        lx, rx = x[:, :, 0], x[:, :, 1]
        lx_si, rx_si = x_si[:, :, 0], x_si[:, :, 1]

        # Non-monotonic Log-einsum-exp trick
        m_lx, _ = torch.max(lx, dim=-1, keepdim=True)  # (-1, num_folds, 1)
        m_rx, _ = torch.max(rx, dim=-1, keepdim=True)  # (-1, num_folds, 1)
        e_lx = lx_si * torch.exp(lx - m_lx)            # (-1, num_folds, num_in_components)
        e_rx = rx_si * torch.exp(rx - m_rx)            # (-1, num_folds, num_in_components)
        # y: (-1, num_folds, num_out_components)
        y = torch.einsum('fkpq,bfp,bfq->bfk', weight, e_lx, e_rx)
        y_si = torch.sign(y.detach())
        y = m_lx + m_rx + safelog(torch.abs(y))
        return y, y_si
