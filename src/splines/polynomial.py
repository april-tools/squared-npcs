import torch
from torch.nn import functional as F


def polyval(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Evaluate polynomials using Horner's method.
    # p: (f, g, ..., k + 1)
    # x: (d, f, g, ...) or (f, g, ...)
    # p and x are broadcastable
    #
    y = p[..., 0]
    for i in range(1, p.shape[-1]):
        y = y * x + p[..., i]  # (d, f, g, ...)
    return y


def polyint(l: torch.Tensor, r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    # Integrate polynomials
    # l: (...)
    # r: (...)
    # p: (..., order + 1)
    order_zero_shape = p.shape[:-1] + (1,)
    indefinite_integrals = torch.concatenate([
        p / torch.arange(p.shape[-1], 0, step=-1, device=p.device),
        torch.zeros(order_zero_shape, dtype=p.dtype, device=p.device)
    ], dim=-1)
    right_hand = polyval(indefinite_integrals, r)  # (...,)
    left_hand = polyval(indefinite_integrals, l)   # (...,)
    return right_hand - left_hand


def cartesian_polymul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # Compute the cartesian product of polynomials.
    # p: (n, k + 1)
    # q: (m, j + 1)
    # o: (m, n, k + j - 1)
    padding = p.shape[-1] - 1
    p = p.unsqueeze(dim=1)
    q = q.unsqueeze(dim=1)
    p = torch.flip(p, dims=(-1,))  # Conv1d implements cross-correlation
    cross_pq = F.conv1d(q, p, padding=padding)  # Compute polynomial product via convolution
    return cross_pq
