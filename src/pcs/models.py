from __future__ import annotations

import abc
from typing import List, Type, Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
from torch import nn

from pcs.layers.input import InputLayer, BornInputLayer
from pcs.layers.input import MultivariateNormalDistribution, BornMultivariateNormalDistribution
from pcs.layers.scope import ScopeLayer, MonotonicScopeLayer, BornScopeLayer
from pcs.layers import ComputeLayer, MonotonicComputeLayer, BornComputeLayer
from pcs.layers.mixture import MonotonicMixtureLayer, BornMixtureLayer
from pcs.layers.tucker import MonotonicTucker2Layer, BornTucker2Layer
from pcs.layers.candecomp import MonotonicCPLayer, BornCPLayer
from region_graph import PartitionNode, RegionNode, RegionGraph

PCS_MODELS = [
    "MonotonicPC", "BornPC"
]


class PC(nn.Module, abc.ABC):
    def __init__(self, num_variables: int, dequantize: bool = False):
        super().__init__()
        self.num_variables = num_variables
        self.dequantize = dequantize
        self.__cache_log_pf = None
        self._device: Optional[Union[int, torch.device]] = None

    def to(self, device: Optional[Union[int, torch.device]] = None, non_blocking: bool = False):
        self._device = device
        return super().to(device=device, non_blocking=non_blocking)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_z = self.log_pf()
        log_y = self.log_score(x)
        log_prob = log_y - log_z
        if self.dequantize:
            log_prob = log_prob - self.num_variables * np.log(256)
        return log_prob

    def log_marginal_prob(self, x: torch.Tensor, mar_mask: torch.Tensor) -> torch.Tensor:
        log_in_z, log_z = self.log_pf(return_input=True)
        log_y = self.log_marginal_score(x, mar_mask, log_in_z=log_in_z)
        log_prob = log_y - log_z
        if self.dequantize:
            log_prob = log_prob - self.num_variables * np.log(256)
        return log_prob

    def invalid_cache_log_pf(self):
        self.__cache_log_pf = None

    def cache_log_pf(self):
        self.__cache_log_pf = self.log_pf(return_input=True)

    def train(self, mode: bool = True) -> PC:
        if self.training and not mode:
            self.cache_log_pf()
        elif not self.training and mode:
            self.invalid_cache_log_pf()
        super().train(mode)
        return self

    def log_pf(
            self,
            *,
            return_input: bool = False
    ) -> Union[torch.Tensor, Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]]:
        if self.__cache_log_pf is None:
            in_log_pf, log_pf = self.eval_log_pf()
        else:
            in_log_pf, log_pf = self.__cache_log_pf
        if return_input:
            return in_log_pf, log_pf
        return log_pf

    @abc.abstractmethod
    def eval_log_pf(self) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        pass

    @abc.abstractmethod
    def log_score(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def log_marginal_score(
            self,
            x: torch.Tensor,
            mar_mask: torch.Tensor,
            log_in_z: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> torch.Tensor:
        pass


class TensorizedPC(PC, abc.ABC):
    def __init__(
            self,
            region_graph: RegionGraph,
            input_layer_cls: Type[InputLayer],
            scope_layer_cls: Type[ScopeLayer],
            compute_layer_cls: Type[ComputeLayer],
            out_mixture_layer_cls: Type[ComputeLayer],
            in_mixture_layer_cls: Type[ComputeLayer],
            input_mixture: bool = False,
            num_classes: int = 1,
            num_input_components: int = 2,
            num_sum_components: int = 2,
            input_layer_kwargs: Optional[dict] = None,
            compute_layer_kwargs: Optional[dict] = None,
            dequantize: bool = False
    ):
        num_variables = region_graph.num_variables
        super().__init__(num_variables, dequantize=dequantize)
        self.region_graph = region_graph
        self.num_classes = num_classes
        self.num_input_components = num_input_components
        self.num_sum_components = num_sum_components
        self.dequantize = dequantize
        self.logit_eps = 1e-6

        # Build the input distribution layer
        rg_layers = region_graph.topological_layers(bottom_up=False)
        if input_layer_kwargs is None:
            input_layer_kwargs = dict()

        self.input_layer = input_layer_cls(rg_layers[0][1], num_input_components, **input_layer_kwargs)

        if compute_layer_kwargs is None:
            compute_layer_kwargs = dict()

        # Build the input scope layer
        if len(rg_layers) > 1:
            self.scope_layer = scope_layer_cls(rg_layers[0][1])
        else:
            self.scope_layer = None

        # Build the input and output mixture layers, if needed
        if input_mixture:
            self.in_mixture = in_mixture_layer_cls(
                rg_layers[0][1],
                num_in_components=num_input_components,
                num_out_components=num_input_components,
                **compute_layer_kwargs
            )
        else:
            self.in_mixture = None

        self.bookkeeping, inner_layers = self._build_layers(
            rg_layers,
            compute_layer_cls,
            compute_layer_kwargs,
            in_mixture_layer_cls,
            num_sum_components,
            num_input_components,
            num_classes=num_classes
        )
        self.layers = nn.ModuleList()
        self.layers.extend(inner_layers)

        if self.input_layer.num_replicas > 1 or len(self.layers) == 0:
            if len(self.layers) == 0:
                num_in_components = self.input_layer.num_components
            else:
                num_in_components = self.input_layer.num_replicas
            mixture_layer_kwargs = compute_layer_kwargs.copy()
            if out_mixture_layer_cls is MonotonicMixtureLayer:
                if 'exp_reparam' in mixture_layer_kwargs:
                    del mixture_layer_kwargs['exp_reparam']
            self.out_mixture = out_mixture_layer_cls(
                rg_layers[-1][1],
                num_in_components=num_in_components,
                num_out_components=self.num_classes,
                **mixture_layer_kwargs)
        else:
            self.out_mixture = None

    @staticmethod
    def _build_layers(
        rg_layers: List[Tuple[List[PartitionNode], List[RegionNode]]],
        compute_layer_cls: Type[ComputeLayer],
        compute_layer_kwargs: Dict[str, Any],
        mixture_layer_cls: Type[ComputeLayer],
        num_sum_components: int,
        num_input_components: int,
        num_classes: int = 1
    ) -> Tuple[List[Tuple[List[int], torch.Tensor]], List[ComputeLayer]]:
        inner_layers: List[ComputeLayer] = []
        bookkeeping: List[Tuple[List[int], torch.Tensor]] = []

        # A dictionary mapping each region node ID to
        #   (i) its index in the corresponding fold, and
        #   (ii) the id of the layer that computes such fold
        #        (0 for the input layer and > 0 for inner layers)
        region_id_fold: Dict[int, Tuple[int, int]] = {}
        for i, region in enumerate(rg_layers[0][1]):
            region_id_fold[region.get_id()] = (i, 0)

        # A list mapping layer ids to the number of folds in the output tensor
        num_folds = [len(rg_layers[0][1])]

        # Build inner layers
        for rg_layer_idx, (lpartitions, lregions) in enumerate(rg_layers[1:], start=1):
            # Gather the input regions of each partition
            input_regions = [sorted(p.inputs) for p in lpartitions]

            # Retrieve which folds need to be concatenated
            input_regions_ids = [list(r.get_id() for r in ins) for ins in input_regions]
            input_layers_ids = [
                list(region_id_fold[i][1] for i in ids) for ids in input_regions_ids
            ]
            unique_layer_ids = list(set(i for ids in input_layers_ids for i in ids))
            cumulative_idx: List[int] = np.cumsum(  # type: ignore[misc]
                [0] + [num_folds[i] for i in unique_layer_ids]
            ).tolist()
            base_layer_idx = dict(zip(unique_layer_ids, cumulative_idx))

            # Build indices
            input_region_indices = []
            for regions in input_regions:
                region_indices = []
                for r in regions:
                    fold_idx, layer_id = region_id_fold[r.get_id()]
                    region_indices.append(base_layer_idx[layer_id] + fold_idx)
                input_region_indices.append(region_indices)
                
            fold_indices = torch.tensor(input_region_indices)
            if fold_indices.shape[1] == 1:
                fold_indices = fold_indices.squeeze(dim=1)

            book_entry = (unique_layer_ids, fold_indices)
            bookkeeping.append(book_entry)

            # Update dictionaries and number of folds
            for i, p in enumerate(lpartitions):
                # Each partition must belong to exactly one region
                assert len(p.outputs) == 1
                out_region = p.outputs[0]
                region_id_fold[out_region.get_id()] = (i, len(inner_layers) + 1)
            num_folds.append(len(lpartitions))

            # Build the actual layer
            num_outputs = num_sum_components if rg_layer_idx < len(rg_layers) - 1 else num_classes
            num_inputs = num_input_components if rg_layer_idx == 1 else num_sum_components

            if all(len(p.inputs) == 1 for p in lpartitions):
                layer = mixture_layer_cls(lregions, num_inputs, num_outputs, **compute_layer_kwargs)
            else:
                assert all(len(p.inputs) == 2 for p in lpartitions)
                layer = compute_layer_cls(lregions, num_inputs, num_outputs, **compute_layer_kwargs)

            inner_layers.append(layer)

        return bookkeeping, inner_layers

    @abc.abstractmethod
    def _eval_layers(self, *args, **kwargs) -> torch.Tensor:
        pass

    def _eval_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dequantize:
            x, ldj = self._logit(x)
        else:
            ldj = torch.zeros(())
        return self.input_layer(x), ldj

    def _logit(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.logit_eps + (1.0 - 2.0 * self.logit_eps) * x
        lx = torch.log(x)
        rx = torch.log(1.0 - x)
        u = lx - rx
        v = lx + rx
        log_det_jacobian = torch.sum(v, dim=1, keepdim=True)
        return u, self.num_variables * np.log(1.0 - 2.0 * self.logit_eps) - log_det_jacobian

    def _unlogit(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = torch.sigmoid(u)
        x = (u - self.logit_eps) / (1.0 - 2.0 * self.logit_eps)
        lu = torch.log(u)
        ru = torch.log(1.0 - u)
        v = lu + ru
        log_det_jacobian = torch.sum(v, dim=1, keepdim=True)
        return x, log_det_jacobian - self.num_variables * np.log(1.0 - 2.0 * self.logit_eps)


class MonotonicPC(TensorizedPC):
    def __init__(
            self,
            region_graph: RegionGraph,
            input_layer_cls: Type[InputLayer],
            compute_layer_cls: Type[MonotonicComputeLayer] = MonotonicCPLayer,
            out_mixture_layer_cls: Type[MonotonicComputeLayer] = MonotonicMixtureLayer,
            in_mixture_layer_cls: Type[MonotonicComputeLayer] = MonotonicMixtureLayer,
            num_components: int = 2,
            **kwargs
    ):
        if compute_layer_cls not in [MonotonicCPLayer, MonotonicTucker2Layer]:
            raise ValueError(f"Invalid compute layer called {compute_layer_cls.__name__}")
        super().__init__(
            region_graph, input_layer_cls, MonotonicScopeLayer, compute_layer_cls,
            out_mixture_layer_cls, in_mixture_layer_cls,
            num_input_components=num_components,
            num_sum_components=num_components,
            **kwargs)

    def _eval_layers(self, x: torch.Tensor) -> torch.Tensor:
        if self.scope_layer is not None:
            x = self.scope_layer(x)

        if self.in_mixture is not None:
            x = self.in_mixture(x)

        layer_outputs: List[torch.Tensor] = [x]
        for layer, (in_layer_ids, fold_idx) in zip(self.layers, self.bookkeeping):
            if len(in_layer_ids) == 1:
                # (B, F, K)
                (in_layer_id,) = in_layer_ids
                inputs = layer_outputs[in_layer_id]
            else:
                # (B, F_1 + ... + F_n, K)
                inputs = torch.cat([layer_outputs[i] for i in in_layer_ids], dim=1)
            inputs = inputs[:, fold_idx]  # inputs: (B, F, H, K)
            outputs = layer(inputs)  # outputs: (B, F, K)
            layer_outputs.append(outputs)

        outputs = layer_outputs[-1]  # (B, F, K)
        if self.out_mixture is not None:
            x = outputs.view(outputs.shape[0], 1, -1)
            outputs = self.out_mixture(x)  # (B, 1, 1)

        return outputs[:, 0]  # (B, 1)

    def eval_log_pf(self) -> Tuple[torch.Tensor, torch.Tensor]:
        log_in_z = self.input_layer.log_pf()
        log_z = self._eval_layers(log_in_z)
        return log_in_z, log_z

    def log_score(self, x: torch.Tensor) -> torch.Tensor:
        x, ldj = self._eval_input(x)
        return self._eval_layers(x) + ldj

    def log_marginal_score(
            self,
            x: torch.Tensor,
            mar_mask: torch.Tensor,
            log_in_z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        if isinstance(self.input_layer, MultivariateNormalDistribution):
            raise NotImplementedError("Marginalization of multivariate input distributions is not supported")
        
        if log_in_z is None:
            if self.training:
                log_in_z = self.input_layer.log_pf()
            else:
                log_in_z, _ = self.log_pf()
        
        x, ldj = self._eval_input(x)
        mar_mask = mar_mask.float().unsqueeze(dim=2).unsqueeze(dim=3)
        x = (1.0 - mar_mask) * x + mar_mask * log_in_z
        x = self._eval_layers(x) + ldj
        
        return x


class BornPC(TensorizedPC):
    def __init__(
            self,
            region_graph: RegionGraph,
            input_layer_cls: Type[BornInputLayer],
            compute_layer_cls: Type[BornComputeLayer] = BornCPLayer,
            out_mixture_layer_cls: Type[ComputeLayer] = MonotonicMixtureLayer,
            in_mixture_layer_cls: Type[BornComputeLayer] = BornMixtureLayer,
            num_components: int = 2,
            **kwargs
    ):
        if compute_layer_cls not in [BornCPLayer, BornTucker2Layer]:
            raise ValueError(f"Invalid compute layer called {compute_layer_cls.__name__}")
        super().__init__(
            region_graph, input_layer_cls, BornScopeLayer, compute_layer_cls,
            out_mixture_layer_cls, in_mixture_layer_cls,
            num_input_components=num_components,
            num_sum_components=num_components,
            **kwargs)

    def _eval_layers(self, x: torch.Tensor, x_si: torch.Tensor, square_mode: bool = False) -> torch.Tensor:
        if self.scope_layer is not None:
            x, x_si = self.scope_layer(x, x_si, square=square_mode)
        
        if self.in_mixture is not None:
            x, x_si = self.in_mixture(x, x_si, square=square_mode)
        layer_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = [(x, x_si)]
        
        for layer, (in_layer_ids, fold_idx) in zip(self.layers, self.bookkeeping):
            if len(in_layer_ids) == 1:
                # (B, F, K)
                (in_layer_id,) = in_layer_ids
                inputs, inputs_si = layer_outputs[in_layer_id]
            else:
                # (B, F_1 + ... + F_n, K)
                inputs = torch.cat([layer_outputs[i][0] for i in in_layer_ids], dim=1)
                inputs_si = torch.cat([layer_outputs[i][1] for i in in_layer_ids], dim=1)
                
            inputs = inputs[:, fold_idx]  # inputs: (B, F, H, K)
            inputs_si = inputs_si[:, fold_idx]
            outputs, outputs_si = layer(inputs, inputs_si, square=square_mode)  # outputs: (B, F, K)
            layer_outputs.append((outputs, outputs_si))
        outputs, outputs_si = layer_outputs[-1]  # (B, F, K)
        
        if self.out_mixture is None:
            outputs = outputs[:, 0]
            return outputs.squeeze(dim=-1) if square_mode else 2 * outputs
        
        if isinstance(self.out_mixture, BornComputeLayer):
            if square_mode:
                # x: (-1, 1, num_components, num_components)
                outputs = outputs.view(outputs.shape[0], 1, outputs.shape[-2], outputs.shape[-1])
                outputs_si = outputs_si.view(outputs_si.shape[0], 1, outputs_si.shape[-2], outputs_si.shape[-1])
            else:
                # x: (-1, 1, num_replicas) or (-1, 1, num_components)
                outputs = outputs.view(outputs.shape[0], 1, -1)
                outputs_si = outputs_si.view(outputs_si.shape[0], 1, -1)
            outputs, _ = self.out_mixture(outputs, outputs_si, square=square_mode)
            outputs = outputs.squeeze(dim=-1) if square_mode else 2 * outputs
            return outputs
        
        x = outputs.view(outputs.shape[0], 1, -1)  # (-1, 1, num_replicas)
        
        if not square_mode:
            x = 2 * x
        x = self.out_mixture(x)  # (B, 1, 1)

        return x.squeeze(dim=-1)

    def eval_log_pf(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        log_in_z, in_z_si = self.input_layer.log_pf()
        log_z = self._eval_layers(log_in_z, in_z_si, square_mode=True)
        return (log_in_z, in_z_si), log_z

    def log_score(self, x: torch.Tensor) -> torch.Tensor:
        (x, x_si), ldj = self._eval_input(x)
        return self._eval_layers(x, x_si) + ldj

    def log_marginal_score(
            self,
            x: torch.Tensor,
            mar_mask: torch.Tensor,
            log_in_z: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        if isinstance(self.input_layer, BornMultivariateNormalDistribution):
            raise NotImplementedError("Marginalization of multivariate input distributions is not supported")
        if log_in_z is None:
            if self.training:
                log_in_z, in_z_si = self.input_layer.log_pf()
            else:
                (log_in_z, in_z_si), _ = self.log_pf()
        else:
            log_in_z, in_z_si = log_in_z
        (x, x_si), ldj = self._eval_input(x)
        x = x.unsqueeze(dim=-2) + x.unsqueeze(dim=-1)
        x_si = x_si.unsqueeze(dim=-2) * x_si.unsqueeze(dim=-1)
        mar_mask = mar_mask.float().unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)
        x = (1.0 - mar_mask) * x + mar_mask * log_in_z
        x_si = (1.0 - mar_mask) * x_si + mar_mask * in_z_si
        return self._eval_layers(x, x_si, square_mode=True) + ldj
