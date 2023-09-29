from __future__ import annotations

import abc
from typing import Tuple, Union, Optional

import torch
from torch import nn

from pcs.initializers import init_params_
from pcs.models import PC
from pcs.utils import safelog


HMM_MODELS = [
    "MonotonicHMM", "BornHMM"
]


class MonotonicHMM(PC, abc.ABC):
    def __init__(
            self,
            vocab_size: int,
            seq_length: int,
            hidden_size: int = 2,
            init_method: str = 'uniform',
            init_scale: float = 1.0
    ):
        assert seq_length > 1
        super().__init__(num_variables=seq_length)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        latent_prior = torch.empty(self.hidden_size)
        init_params_(latent_prior, init_method, init_scale=init_scale)
        self.latent_prior = nn.Parameter(torch.log(latent_prior), requires_grad=True)

        latent_conds = torch.empty(self.hidden_size, self.hidden_size)
        init_params_(latent_conds, init_method, init_scale=init_scale)
        self.latent_conds = nn.Parameter(torch.log(latent_conds), requires_grad=True)

        emission_conds = torch.empty(self.hidden_size, self.vocab_size)
        init_params_(emission_conds, init_method, init_scale=init_scale)
        self.emission_conds = nn.Parameter(torch.log(emission_conds), requires_grad=True)

    def eval_log_pf(self) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        log_pf = torch.zeros(size=(1, 1), device=self._device)
        log_in_pf = torch.zeros(size=(1, self.hidden_size), device=self._device)
        return log_in_pf, log_pf

    def _latent_prior(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, hidden_size)
        # latent_prior: (hidden_size,)
        latent_prior = torch.softmax(self.latent_prior, dim=0)
        m_x, _ = torch.max(x, dim=1, keepdim=True)
        e_x = torch.exp(x - m_x)
        x = torch.sum(latent_prior * e_x, dim=1, keepdim=True)
        x = m_x + safelog(x)
        return x

    def _latent_conds(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, hidden_size)
        # latent_conds: (hidden_size, hidden_size)
        latent_conds = torch.softmax(self.latent_conds, dim=1)
        m_x, _ = torch.max(x, dim=1, keepdim=True)
        e_x = torch.exp(x - m_x)
        x = torch.mm(e_x, latent_conds.T)
        x = m_x + safelog(x)
        return x

    def log_score(self, x: torch.Tensor) -> torch.Tensor:
        emission_conds = torch.log_softmax(self.emission_conds, dim=1)
        y = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        for i in range(self.num_variables - 1, -1, -1):
            if i != self.num_variables - 1:
                y = self._latent_conds(y)
            zi = torch.arange(self.hidden_size, device=y.device).unsqueeze(dim=0)
            w = emission_conds[zi, x[:, i].unsqueeze(dim=-1)]
            y = y + w
        y = self._latent_prior(y)
        return y

    def log_marginal_score(
            self,
            x: torch.Tensor,
            mar_mask: torch.Tensor,
            log_in_z: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> torch.Tensor:
        raise NotImplementedError()


class BornHMM(PC, abc.ABC):
    def __init__(
            self,
            vocab_size: int,
            seq_length: int,
            hidden_size: int = 2,
            init_method: str = 'normal',
            init_scale: float = 1.0
    ):
        assert seq_length > 1
        super().__init__(num_variables=seq_length)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        latent_prior = torch.empty(self.hidden_size)
        init_params_(latent_prior, init_method, init_scale=init_scale)
        self.latent_prior = nn.Parameter(latent_prior, requires_grad=True)

        latent_conds = torch.empty(self.hidden_size, self.hidden_size)
        init_params_(latent_conds, init_method, init_scale=init_scale)
        self.latent_conds = nn.Parameter(latent_conds, requires_grad=True)

        emission_conds = torch.empty(self.hidden_size, self.vocab_size)
        init_params_(emission_conds, init_method, init_scale=init_scale)
        self.emission_conds = nn.Parameter(emission_conds, requires_grad=True)

    def eval_log_pf(self) -> Tuple[Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        log_pf, _ = self._eval_layers_normalize()
        return None, log_pf

    def _latent_prior(self, x: torch.Tensor, x_si: torch.Tensor, square: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if square:
            # x: (batch_size, hidden_size, hidden_size)
            # self.latent_prior: (hidden_size,)
            m_x, _ = torch.max(x, dim=2, keepdim=True)  # (batch_size, hidden_size, 1)
            x = x_si * torch.exp(x - m_x)
            x = torch.sum(self.latent_prior * x, dim=2)
            x_si = torch.sign(x.detach())
            x = m_x.squeeze(dim=2) + safelog(torch.abs(x))  # (batch_size, hidden_size)
            m_x, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size, 1)
            x = x_si * torch.exp(x - m_x)
            x = torch.sum(self.latent_prior * x, dim=1, keepdim=True)
            x_si = torch.sign(x.detach())
            x = m_x + safelog(torch.abs(x))  # (batch_size, 1)
            return x, x_si
        # x: (batch_size, hidden_size)
        # self.latent_prior: (hidden_size,)
        m_x, _ = torch.max(x, dim=1, keepdim=True)
        x = x_si * torch.exp(x - m_x)
        y = torch.sum(self.latent_prior * x, dim=1, keepdim=True)
        y_si = torch.sign(y.detach())
        y = safelog(torch.abs(y)) + m_x
        return y, y_si

    def _latent_conds(self, x: torch.Tensor, x_si: torch.Tensor, square: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if square:
            # x: (batch_size, hidden_size, hidden_size)
            # self.latent_conds: (hidden_size, hidden_size)
            m_x, _ = torch.max(x, dim=2, keepdim=True)  # (batch_size, hidden_size, 1)
            x = x_si * torch.exp(x - m_x)
            x = torch.einsum('pi,bji->bpj', self.latent_conds, x)
            x_si = torch.sign(x.detach())
            x = m_x.permute(0, 2, 1) + safelog(torch.abs(x))  # (batch_size, hidden_size, hidden_size)
            m_x, _ = torch.max(x, dim=2, keepdim=True)  # (batch_size, hidden_size, 1)
            x = x_si * torch.exp(x - m_x)
            x = torch.einsum('qj,bpj->bpq', self.latent_conds, x)
            x_si = torch.sign(x.detach())
            x = m_x + safelog(torch.abs(x))  # (batch_size, hidden_size, hidden_size)
            return x, x_si
        # x: (batch_size, hidden_size)
        # self.latent_conds: (hidden_size, hidden_size)
        m_x, _ = torch.max(x, dim=1, keepdim=True)
        x = x_si * torch.exp(x - m_x)
        y = torch.einsum('ij,bj->bi', self.latent_conds, x)
        y_si = torch.sign(y.detach())
        y = safelog(torch.abs(y)) + m_x
        return y, y_si

    def _emission_conds(self, x: torch.Tensor, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        zi = torch.arange(self.hidden_size, device=x.device).unsqueeze(dim=0)
        w = self.emission_conds[zi, x[:, i].unsqueeze(dim=-1)]
        w_si = torch.sign(w.detach())
        w = safelog(torch.abs(w))
        return w, w_si

    def _emission_conds_normalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w_si = torch.sign(self.emission_conds.detach())
        w = safelog(torch.abs(self.emission_conds))
        m_w, _ = torch.max(w, dim=1, keepdim=True)
        e_w = w_si * torch.exp(w - m_w)
        z = torch.mm(e_w, e_w.T)
        z_si = torch.sign(z.detach())
        z = m_w + m_w.T + safelog(torch.abs(z))
        return z.unsqueeze(dim=0), z_si.unsqueeze(dim=0)

    def _eval_layers(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        y_si = torch.ones_like(y)
        for i in range(self.num_variables - 1, -1, -1):
            if i != self.num_variables - 1:
                y, y_si = self._latent_conds(y, y_si)
            w, w_si = self._emission_conds(x, i)
            y = y + w
            y_si = y_si * w_si
        y, y_si = self._latent_prior(y, y_si)
        return y, y_si

    def _eval_layers_normalize(self):
        w, w_si = self._emission_conds_normalize()
        y = torch.zeros(1, self.hidden_size, self.hidden_size, device=self._device)
        y_si = torch.ones_like(y)
        for i in range(self.num_variables - 1, -1, -1):
            if i != self.num_variables - 1:
                y, y_si = self._latent_conds(y, y_si, square=True)
            y = y + w
            y_si = y_si * w_si
        y, y_si = self._latent_prior(y, y_si, square=True)
        return y, y_si

    def log_score(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self._eval_layers(x)
        return 2.0 * y

    def log_marginal_score(
            self,
            x: torch.Tensor,
            mar_mask: torch.Tensor,
            log_in_z: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> torch.Tensor:
        raise NotImplementedError()
