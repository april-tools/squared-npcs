import abc
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from pcs.utils import retrieve_default_dtype
from region_graph import RegionNode


class ScopeLayer(nn.Module, abc.ABC):
    def __init__(self, rg_nodes: List[RegionNode]):
        super().__init__()
        scope = ScopeLayer.__build_scope(rg_nodes)
        self.register_buffer('scope', torch.from_numpy(scope))

    @staticmethod
    def __build_scope(rg_nodes: List[RegionNode]) -> np.ndarray:
        replica_indices = set(n.get_replica_idx() for n in rg_nodes)
        num_replicas = len(replica_indices)
        assert replica_indices == set(
            range(num_replicas)
        ), "Replica indices should be consecutive, starting with 0."
        num_variables = len(set(v for n in rg_nodes for v in n.scope))
        scope = np.zeros(shape=(len(rg_nodes), num_variables, num_replicas),
                         dtype=retrieve_default_dtype(numpy=True))
        for i, n in enumerate(rg_nodes):
            scope[i, list(n.scope), n.get_replica_idx()] = 1.0
        return scope


class MonotonicScopeLayer(ScopeLayer):
    def __init__(self, rg_nodes: List[RegionNode]):
        super().__init__(rg_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_vars, num_replicas, num_components)
        # y: (-1, num_folds, num_components)
        return torch.einsum('bvri,fvr->bfi', x, self.scope)


class BornScopeLayer(ScopeLayer):
    def __init__(self, rg_nodes: List[RegionNode]):
        super().__init__(rg_nodes)

    def forward(self, x: torch.Tensor, x_si: torch.Tensor, square: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if square:
            # x: (-1, num_vars, num_replicas, num_components, num_components)
            # y: (-1, num_folds, num_components, num_components)
            x = torch.einsum('bvrij,fvr->bfij', x, self.scope)
            x_si[x_si > 0.0] = 0.0
            x_si = torch.einsum('bvrij,fvr->bfij', x_si, self.scope)
            x_si = 2.0 * torch.fmod(x_si, 2) + 1.0  # (trick: compute signs by performing an einsum)
            return x, x_si

        # x: (-1, num_vars, num_replicas, num_components)
        # y: (-1, num_folds, num_components)
        x = torch.einsum('bvri,fvr->bfi', x, self.scope)
        x_si[x_si > 0.0] = 0.0
        x_si = torch.einsum('bvri,fvr->bfi', x_si, self.scope)
        x_si = 2.0 * torch.fmod(x_si, 2) + 1.0  # (trick: compute signs by performing an einsum)
        return x, x_si
