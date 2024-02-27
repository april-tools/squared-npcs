import abc
from typing import Tuple, Union, List

import numpy as np
import torch
from torch import nn, distributions as ds
from torch.distributions.multivariate_normal import _batch_mahalanobis as batch_mahalanobis

from pcs.initializers import init_params_
from region_graph import RegionNode
from splines.bsplines import splines_uniform_polynomial, basis_polyval, basis_polyint, integrate_cartesian_basis, \
    least_squares_basis
from pcs.utils import retrieve_default_dtype, ohe, safelog, log_binomial


class InputLayer(nn.Module, abc.ABC):
    def __init__(self, rg_nodes: List[RegionNode], num_components: int, **kwargs):
        super().__init__()
        self.rg_nodes = rg_nodes
        self.num_components = num_components
        replica_indices = set(n.get_replica_idx() for n in rg_nodes)
        self.num_replicas = len(replica_indices)
        assert replica_indices == set(
            range(self.num_replicas)
        ), "Replica indices should be consecutive, starting with 0."
        self.num_variables = len(set(v for n in rg_nodes for v in n.scope))

    @abc.abstractmethod
    def log_pf(self) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass


class MonotonicInputLayer(InputLayer, abc.ABC):
    @abc.abstractmethod
    def log_pf(self) -> torch.Tensor:
        pass


class BornInputLayer(InputLayer, abc.ABC):
    @abc.abstractmethod
    def log_pf(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class MonotonicEmbeddings(MonotonicInputLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            num_states: int = 2,
            init_method: str = 'dirichlet',
            init_scale: float = 1.0
    ):
        super().__init__(rg_nodes, num_components)
        self.num_states = num_states
        weight = torch.empty(self.num_variables, self.num_replicas, self.num_components, num_states)
        init_params_(weight, init_method, init_scale=init_scale)
        self.weight = nn.Parameter(torch.log(weight), requires_grad=True)
        self._ohe = num_states <= 256

    def log_pf(self) -> torch.Tensor:
        # log_z: (1, num_vars, num_replicas, num_components)
        log_z = torch.logsumexp(self.weight, dim=-1).unsqueeze(dim=0)
        return log_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_vars)
        # self.weight: (num_vars, num_comps, num_states)
        if self._ohe:
            x = ohe(x, self.num_states)  # (-1, num_vars, num_states)
            # log_y: (-1, num_vars, num_replicas, num_components)
            log_y = torch.einsum('bvd,vrid->bvri', x, self.weight)
        else:
            weight = self.weight.permute(0, 3, 1, 2)
            log_y = weight[torch.arange(weight.shape[0], device=x.device), x]
        return log_y


class BornEmbeddings(BornInputLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            num_states: int = 2,
            init_method: str = 'normal',
            init_scale: float = 1.0,
            exp_reparam: bool = False,
            l2norm: bool = False
    ):
        assert not exp_reparam or not l2norm, "Only one between --exp-reparam and --l2norm can be set true"
        super().__init__(rg_nodes, num_components)
        self.num_states = num_states
        weight = torch.empty(self.num_variables, self.num_replicas, self.num_components, num_states)
        init_params_(weight, init_method, init_scale=init_scale)
        if exp_reparam:
            weight = torch.log(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.exp_reparam = exp_reparam
        self.l2norm = l2norm
        self._ohe = num_states <= 256

    def log_pf(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.exp_reparam:
            weight = torch.exp(self.weight)
        elif self.l2norm:
            weight = self.weight / torch.linalg.vector_norm(self.weight, ord=2, dim=2, keepdim=True)
        else:
            weight = self.weight

        w_si = torch.sign(weight.detach())  # (num_variables, num_replicas, num_components, num_states)
        w = safelog(torch.abs(weight))      # (num_variables, num_replicas, num_components, num_states)
        m_w, _ = torch.max(w, dim=3, keepdim=True)  # (num_variables, num_replicas, num_components, 1)
        e_w = w_si * torch.exp(w - m_w)
        z = torch.einsum('vrid,vrjd->vrij', e_w, e_w)  # (num_variables, num_replicas, num_components, num_components)
        z_si = torch.sign(z.detach())
        z = m_w + m_w.permute(0, 1, 3, 2) + safelog(torch.abs(z))
        return z.unsqueeze(dim=0), z_si.unsqueeze(dim=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.exp_reparam:
            weight = torch.exp(self.weight)
        elif self.l2norm:
            weight = self.weight / torch.linalg.vector_norm(self.weight, ord=2, dim=2, keepdim=True)
        else:
            weight = self.weight

        # x: (-1, num_vars)
        # self.weight: (num_vars, num_comps, num_states)
        if self._ohe:
            x = ohe(x, self.num_states)  # (-1, num_vars, num_states)
            # (-1, num_vars, num_replicas, num_components)
            w = torch.einsum('bvd,vrid->bvri', x, weight)
            # Split into log absolute value and sign components
            w_si = torch.sign(w.detach())
            w = safelog(torch.abs(w))
        else:
            weight = weight.permute(0, 3, 1, 2)
            w = weight[torch.arange(weight.shape[0], device=x.device), x]
            w_si = torch.sign(w.detach())
            w = safelog(torch.abs(w))
        return w, w_si


class MonotonicBinaryEmbeddings(MonotonicEmbeddings):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            init_method: str = 'dirichlet',
            init_scale: float = 1.0
    ):
        super().__init__(
            rg_nodes, num_components=num_components, num_states=2, init_method=init_method, init_scale=init_scale)


class BornBinaryEmbeddings(BornEmbeddings):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            init_method: str = 'normal',
            init_scale: float = 1.0,
            exp_reparam: bool = False
    ):
        super().__init__(
            rg_nodes, num_components=num_components, num_states=2,
            init_method=init_method, init_scale=init_scale, exp_reparam=exp_reparam)


class MonotonicBinomial(MonotonicInputLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            num_states: int = 2
    ):
        super().__init__(rg_nodes, num_components)
        self.num_states = num_states
        self.total_count = num_states - 1
        weight = torch.empty(self.num_variables, self.num_replicas, self.num_components)
        init_params_(weight, 'uniform', init_loc=0.0, init_scale=1.0)
        self.weight = nn.Parameter(torch.logit(weight), requires_grad=True)
        self.register_buffer('log_bcoeffs', torch.tensor([
            log_binomial(self.total_count, k) for k in range(num_states)
        ], dtype=torch.get_default_dtype()))
        self._log_sigmoid = nn.LogSigmoid()

    def log_pf(self) -> torch.Tensor:
        return torch.zeros(
            1, self.num_variables, self.num_replicas, self.num_components,
            device=self.weight.device, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_vars, 1, 1)
        # self.weight: (num_vars, num_components, num_states)
        x = x.unsqueeze(dim=2).unsqueeze(dim=3)

        # log_coeffs: (-1, num_vars, 1, 1)
        log_bcoeffs = self.log_bcoeffs[x]

        # log_success: (-1, num_vars, num_replicas, num_components)
        # log_failure: (-1, num_vars, num_replicas, num_components)
        log_probs = self._log_sigmoid(self.weight).unsqueeze(dim=0)
        log_success = x * log_probs
        log_failure = (self.total_count - x) * torch.log1p(-torch.exp(log_probs))

        # log_y: (-1, num_vars, num_replicas, num_components)
        log_y = log_bcoeffs + log_success + log_failure
        return log_y


class BornBinomial(BornInputLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            num_states: int = 2
    ):
        super().__init__(rg_nodes, num_components)
        self.num_states = num_states
        self.total_count = num_states - 1
        weight = torch.empty(self.num_variables, self.num_replicas, self.num_components)
        init_params_(weight, 'uniform', init_loc=0.0, init_scale=1.0)
        self.weight = nn.Parameter(torch.logit(weight), requires_grad=True)
        self.register_buffer('log_bcoeffs', torch.tensor([
            log_binomial(self.total_count, k) for k in range(num_states)
        ], dtype=torch.get_default_dtype()))
        self._log_sigmoid = nn.LogSigmoid()

    def log_pf(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # counts, log_bcoeefs: (num_states, 1, 1, 1)
        counts = torch.arange(self.num_states, device=self.weight.device)\
            .unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
        log_bcoeffs = self.log_bcoeffs.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)

        # log_success: (num_states, num_vars, num_replicas, num_components)
        # log_failure: (num_states, num_vars, num_replicas, num_components)
        log_success_probs = self._log_sigmoid(self.weight).unsqueeze(dim=0)
        log_success = counts * log_success_probs
        log_failure = (self.total_count - counts) * torch.log1p(-torch.exp(log_success_probs))
        log_probs = log_bcoeffs + log_success + log_failure

        # log_z: (1, num_variables, num_replicas, num_components, num_components)
        m_lp, _ = torch.max(log_probs, dim=0, keepdim=True)  # (1, num_vars, num_replicas, num_components)
        e_lp = torch.exp(log_probs - m_lp)                   # (num_states, num_vars, num_replicas, num_components)
        z = torch.einsum('dvri,dvrj->vrij', e_lp, e_lp)
        log_z = safelog(z.unsqueeze(dim=0)) + m_lp.unsqueeze(dim=-2) + m_lp.unsqueeze(dim=-1)
        return log_z, torch.ones_like(log_z, requires_grad=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (-1, num_vars, 1, 1)
        # self.weight: (num_vars, num_components)
        x = x.unsqueeze(dim=2).unsqueeze(dim=3)

        # log_coeffs: (-1, num_vars, 1, 1)
        log_bcoeffs = self.log_bcoeffs[x]

        # log_success: (-1, num_vars, num_components)
        # log_failure: (-1, num_vars, num_components)
        log_success_probs = self._log_sigmoid(self.weight).unsqueeze(dim=0)
        log_success = x * log_success_probs
        log_failure = (self.total_count - x) * torch.log1p(-torch.exp(log_success_probs))

        # log_probs: (-1, num_vars, num_replicas, num_components)
        log_probs = log_bcoeffs + log_success + log_failure
        return log_probs, torch.ones_like(log_probs, requires_grad=False)


class NormalDistribution(MonotonicInputLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            init_scale: float = 1.0
    ):
        super().__init__(rg_nodes, num_components)
        # Initialize mean
        mu = torch.empty(1, self.num_variables, self.num_replicas, self.num_components)
        init_params_(mu, 'normal', init_scale=1.0)
        self.mu = nn.Parameter(mu, requires_grad=True)
        # Initialize diagonal covariance matrix
        log_sigma = torch.empty(1, self.num_variables, self.num_replicas, self.num_components)
        init_params_(log_sigma, 'normal', init_loc=0.0, init_scale=init_scale)
        self.log_sigma = nn.Parameter(log_sigma)
        self.eps = 1e-5

    def log_pf(self) -> torch.Tensor:
        return torch.zeros(
            1, self.num_variables, self.num_replicas, self.num_components,
            device=self.mu.device, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_variables) -> (-1, num_variables, 1, 1)
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
        scale = torch.exp(self.log_sigma) + self.eps
        # log_prob: (-1, num_vars, num_replicas, num_components)
        log_prob = ds.Normal(loc=self.mu, scale=scale).log_prob(x)
        return log_prob


class MultivariateNormalDistribution(InputLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            init_scale: float = 1.0
    ):
        super().__init__(rg_nodes, num_components)
        if self.num_replicas > 1 or len(rg_nodes) > 1:
            raise NotImplementedError("Multivariate distributions only implemented for single-region networks")
        # Initialize mean
        mu = torch.empty(1, self.num_replicas, self.num_components, self.num_variables)
        init_params_(mu, 'normal', init_scale=1.0)
        self.mu = nn.Parameter(mu, requires_grad=True)
        # Initialize weight for parametrizing the full covariance matrix
        weight = torch.empty(1, self.num_replicas, self.num_components, self.num_variables, self.num_variables)
        init_params_(weight, 'normal', init_scale=init_scale)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.register_buffer('eps_eye', 1e-5 * torch.eye(self.num_variables))

    def log_pf(self) -> torch.Tensor:
        return torch.zeros(
            1, self.num_replicas, len(self.rg_nodes), self.num_components,
            device=self.mu.device, requires_grad=False)

    def __cholesky(self):
        weight = self.weight.squeeze(dim=0)
        pos_semidef_matrix = torch.matmul(weight, weight.permute(0, 1, 3, 2))
        return torch.linalg.cholesky(pos_semidef_matrix + self.eps_eye)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_variables) -> (-1, 1, 1, num_variables)
        x = x.unsqueeze(dim=1).unsqueeze(dim=1)
        scale_tril = self.__cholesky()
        # log_prob: (-1, num_replicas, 1, num_components)
        log_prob = ds.MultivariateNormal(loc=self.mu, scale_tril=scale_tril).log_prob(x)
        log_prob = log_prob.unsqueeze(dim=1)
        return log_prob


class BornNormalDistribution(BornInputLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            init_scale: float = 1.0
    ):
        super().__init__(rg_nodes, num_components)
        self.log_two_pi = np.log(2.0 * np.pi)
        # Initialize mean
        mu = torch.empty(1, self.num_variables, self.num_replicas, self.num_components)
        init_params_(mu, 'normal', init_scale=1.0)
        self.mu = nn.Parameter(mu, requires_grad=True)
        # Initialize diagonal covariance matrix
        log_sigma = torch.empty(1, self.num_variables, self.num_replicas, self.num_components)
        init_params_(log_sigma, 'normal', init_loc=0.0, init_scale=init_scale)
        self.log_sigma = nn.Parameter(log_sigma, requires_grad=True)
        self.eps = 1e-5

    def log_pf(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu
        log_sigma = self.log_sigma
        # log_sum_cov, sq_norm_dist: (1, num_variables, num_replicas, num_components, num_components)
        log_cov = 2.0 * torch.log(torch.exp(log_sigma) + self.eps)
        m_s, _ = torch.max(log_cov, dim=-1, keepdim=True)
        cov = torch.exp(log_cov - m_s)
        log_sum_cov = m_s.unsqueeze(dim=-1) + safelog(cov.unsqueeze(dim=-2) + cov.unsqueeze(dim=-1))
        log_sq_dif_mu = 2.0 * safelog(torch.abs(mu.unsqueeze(dim=-2) - mu.unsqueeze(dim=-1)))
        sq_norm_dist = torch.exp(log_sq_dif_mu - log_sum_cov)
        # log_z: (1, num_variables, num_replicas, num_components, num_components)
        log_z = -0.5 * (self.log_two_pi + log_sum_cov + sq_norm_dist)
        return log_z, torch.ones_like(log_z, requires_grad=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (-1, num_variables) -> (-1, num_variables, 1, 1)
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
        scale = torch.exp(self.log_sigma) + self.eps
        # log_prob: (-1, num_variables, num_replicas, num_components)
        log_prob = ds.Normal(loc=self.mu, scale=scale).log_prob(x)
        return log_prob, torch.ones_like(log_prob, requires_grad=False)


class BornMultivariateNormalDistribution(BornInputLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            init_scale: float = 1.0
    ):
        super().__init__(rg_nodes, num_components)
        if self.num_replicas > 1 or len(rg_nodes) > 1:
            raise NotImplementedError("Multivariate distributions only implemented for single-region networks")
        self.log_two_pi = np.log(2.0 * np.pi)
        # Initialize mean
        mu = torch.empty(1, self.num_replicas, self.num_components, self.num_variables)
        init_params_(mu, 'normal', init_scale=1.0)
        self.mu = nn.Parameter(mu, requires_grad=True)
        # Initialize weight for parametrizing the full covariance matrix
        weight = torch.empty(1, self.num_replicas, self.num_components, self.num_variables, self.num_variables)
        init_params_(weight, 'normal', init_scale=init_scale)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.register_buffer('eps_eye', 1e-5 * torch.eye(self.num_variables))

    def __covariance(self):
        weight = self.weight
        pos_semidef_matrix = torch.einsum('brivx,briux->brivu', weight, weight)
        return pos_semidef_matrix + self.eps_eye

    def __cholesky(self):
        return torch.linalg.cholesky(self.__covariance())

    def log_pf(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # mu_a: (1, num_replicas, 1, num_components, num_variables)
        # mu_b: (1, num_replicas, num_components, 1, num_variables)
        mu = self.mu
        mu_a, mu_b = mu.unsqueeze(dim=-3), mu.unsqueeze(dim=-2)
        # cov_a: (1, num_replicas, 1, num_components, num_variables, num_variables)
        # cov_b: (1, num_replicas, num_components, 1, num_variables, num_variables)
        cov = self.__covariance()
        cov_a, cov_b = cov.unsqueeze(dim=-4), cov.unsqueeze(dim=-3)
        pairwise_cov = cov_a + cov_b
        pairwise_scale_tril = torch.linalg.cholesky(pairwise_cov)

        # sq_norm_dist: (1, num_replicas, num_components, num_components)
        sq_norm_dist = batch_mahalanobis(pairwise_scale_tril, mu_a - mu_b)
        # log_det_cov: (1, num_replicas, num_components, num_components)
        log_det_cov = torch.sum(safelog(pairwise_scale_tril.diagonal(dim1=-2, dim2=-1)), dim=-1)

        # log_z: (1, num_replicas, 1, num_components, num_components)
        log_z = -0.5 * (self.mu.shape[-1] * self.log_two_pi + sq_norm_dist) - log_det_cov
        log_z = log_z.unsqueeze(dim=-3)
        return log_z, torch.ones_like(log_z, requires_grad=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (-1, num_variables) -> (-1, 1, 1, num_variables)
        x = x.unsqueeze(dim=1).unsqueeze(dim=1)
        scale_tril = self.__cholesky()
        log_prob = ds.MultivariateNormal(loc=self.mu, scale_tril=scale_tril).log_prob(x)
        log_prob = log_prob.unsqueeze(dim=1)
        return log_prob, torch.ones_like(log_prob, requires_grad=False)


class MonotonicBSplines(MonotonicInputLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            order: int = 1,
            num_knots: int = 3,
            interval: Tuple[float, float] = (0.0, 1.0),
            init_method: str = 'dirichlet',
            init_scale: float = 1.0
    ):
        super().__init__(rg_nodes, num_components)
        self.order = order
        self.num_knots = num_knots
        self.interval = interval

        # Construct the basis functions (as polynomials) and the knots
        knots, polynomials = splines_uniform_polynomial(order, num_knots, interval=interval)
        numpy_dtype = retrieve_default_dtype(numpy=True)
        knots = knots.astype(numpy_dtype, copy=False)
        polynomials = polynomials.astype(numpy_dtype, copy=False)
        self.register_buffer('knots', torch.from_numpy(knots))
        self.register_buffer('polynomials', torch.from_numpy(polynomials))
        self.register_buffer('_integral_basis', basis_polyint(self.knots, self.polynomials))

        # Initialize the coefficients (in log-space) of the splines (along replicas and components dimensions)
        weight = torch.empty(self.num_variables, self.num_replicas, self.num_components, self.num_knots)
        init_params_(weight, init_method, init_scale=init_scale)
        self.weight = nn.Parameter(torch.log(weight), requires_grad=True)

    @torch.no_grad()
    def least_squares_fit(self, data: torch.Tensor, batch_size: int = 1, noise: float = 5e-2):
        coeffs = least_squares_basis(
            self.knots, self.polynomials, data,
            num_replicas=self.num_replicas, num_components=self.num_components,
            batch_size=batch_size, noise=noise
        )
        coeffs = safelog(coeffs)
        self.weight.data.copy_(coeffs)

    def log_pf(self) -> torch.Tensor:
        sint = self._integral_basis
        # log_z: (1, num_variables, num_replicas, num_components)
        log_z = torch.logsumexp(self.weight + safelog(sint), dim=-1)  # (num_variables, num_replicas, num_components)
        log_z = log_z.unsqueeze(dim=0)
        return log_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (-1, num_variables)
        y = basis_polyval(self.knots, self.polynomials, x)   # (-1, num_variables, num_knots)
        log_y = safelog(y)
        m_y, _ = torch.max(log_y, dim=-1, keepdim=True)
        e_y = torch.exp(log_y - m_y)
        e_w = torch.exp(self.weight)
        # log_y: (-1, num_variables, num_replicas, num_components)
        y = torch.einsum('vrik,bvk->bvri', e_w, e_y)
        log_y = m_y.unsqueeze(dim=-1) + safelog(y)
        return log_y


class BornBSplines(BornInputLayer):
    def __init__(
            self,
            rg_nodes: List[RegionNode],
            num_components: int = 2,
            order: int = 1,
            num_knots: int = 3,
            interval: Tuple[float, float] = (0.0, 1.0),
            init_method: str = 'normal',
            init_scale: float = 1.0,
            exp_reparam: bool = False
    ):
        super().__init__(rg_nodes, num_components)
        self.order = order
        self.num_knots = num_knots
        self.interval = interval

        # Construct the basis functions (as polynomials) and the knots
        knots, polynomials = splines_uniform_polynomial(order, num_knots, interval=interval)
        numpy_dtype = retrieve_default_dtype(numpy=True)
        knots = knots.astype(numpy_dtype, copy=False)
        polynomials = polynomials.astype(numpy_dtype, copy=False)
        self.register_buffer('knots', torch.from_numpy(knots))
        self.register_buffer('polynomials', torch.from_numpy(polynomials))
        self.register_buffer('_integral_cartesian_basis', integrate_cartesian_basis(self.knots, self.polynomials))

        # Initialize the coefficients (in log-space) of the splines (along replicas and components dimensions)
        weight = torch.empty(self.num_variables, self.num_replicas, self.num_components, self.num_knots)
        init_params_(weight, init_method, init_scale=init_scale)
        if exp_reparam:
            weight = torch.log(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.exp_reparam = exp_reparam

    @torch.no_grad()
    def least_squares_fit(self, data: torch.Tensor, batch_size: int = 1, noise: float = 5e-2):
        coeffs = least_squares_basis(
            self.knots, self.polynomials, data,
            num_replicas=self.num_replicas, num_components=self.num_components,
            batch_size=batch_size, noise=noise
        )
        if self.exp_reparam:
            coeffs = safelog(coeffs)
        self.weight.data.copy_(coeffs)

    def log_pf(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = torch.exp(self.weight) if self.exp_reparam else self.weight

        # sint: (num_knots, num_knots)
        sint = self._integral_cartesian_basis
        z = torch.einsum('kl,vrik->lvri', sint, weight)
        z = torch.einsum('lvri,vrjl->vrij', z, weight)
        # log_z, z_si: (1, num_variables, num_replicas, num_components, num_components)
        z_si = torch.sign(z.detach()).unsqueeze(dim=0)
        log_z = safelog(torch.abs(z)).unsqueeze(dim=0)
        return log_z, z_si

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = torch.exp(self.weight) if self.exp_reparam else self.weight

        # x: (-1, num_variables)
        y = basis_polyval(self.knots, self.polynomials, x)   # (-1, num_variables, num_knots)
        # y: (-1, num_variables, num_replicas, num_components)
        y = torch.einsum('bvk,vrik->bvri', y, weight)
        y_si = torch.sign(y.detach())
        log_y = safelog(torch.abs(y))
        # log_y: (-1, num_variables, num_replicas, num_components)
        return log_y, y_si
