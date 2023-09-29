from typing import Tuple

import numpy as np
import torch

from splines.polynomial import polyval, polyint, cartesian_polymul


def least_squares_basis(
        knots: torch.Tensor,
        polynomials: torch.Tensor,
        data: torch.Tensor,
        num_replicas: int = 1,
        num_components: int = 1,
        batch_size: int = 1,
        noise: float = 5e-2
) -> torch.Tensor:
    assert polynomials.shape[1] == polynomials.shape[2]
    assert knots.shape[0] == polynomials.shape[0]
    assert knots.shape[1] == polynomials.shape[1]
    assert knots.shape[2] == 2
    assert data.shape[0] > 0 and data.shape[1] > 0
    assert batch_size > 0 and noise > 0.0
    # knots: (num_knots, order + 1, 2)
    # polynomials (num_knots, order + 1, order + 1)
    # data: (num_samples, num_variables)

    b = 0.0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(knots.device)  # (batch, num_variables)
        batch_basis = basis_polyval(knots, polynomials, batch)  # (<= batch_size, num_variables, num_knots)
        b = b + torch.sum(batch_basis, dim=0)
    # b: (num_variables, num_knots, 1)
    b = 2.0 * b.unsqueeze(dim=-1) / len(data)
    # A: (1, num_knots, num_knots)
    A = integrate_cartesian_basis(knots, polynomials).unsqueeze(dim=0)
    x = torch.linalg.lstsq(A, b).solution  # (num_variables, num_knots, 1)
    x = x.squeeze(dim=-1)  # (num_variables, num_knots)

    # Add artificial noise
    g = torch.randn([x.shape[0], num_replicas, num_components, x.shape[1]], device=x.device)
    x = x.unsqueeze(dim=-2).unsqueeze(dim=-2)
    r = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    x = noise * r * g + x

    return x


def integrate_cartesian_basis(knots: torch.Tensor, polynomials: torch.Tensor) -> torch.Tensor:
    assert polynomials.shape[1] == polynomials.shape[2]
    assert knots.shape[0] == polynomials.shape[0]
    assert knots.shape[1] == polynomials.shape[1]
    assert knots.shape[2] == 2

    prod_basis_polynomials = torch.vmap(
        torch.vmap(cartesian_polymul, in_dims=(0, None)), in_dims=(None, 0)
    )(polynomials, polynomials)  # (num_knots, num_knots, order + 1, order + 1, 2 * order + 1)

    # knots_a: (num_knots, 1,   order + 1, 1,   2)
    # knots_b: (1, num_knots,   1, order + 1,   2)
    knots_a = knots.unsqueeze(dim=0).unsqueeze(dim=2)
    knots_b = knots.unsqueeze(dim=1).unsqueeze(dim=3)
    # knots_left, knots_right: (num_knots, num_knots, order + 1, order + 1)
    knots_left = torch.maximum(knots_a[..., 0], knots_b[..., 0])
    knots_right = torch.minimum(knots_a[..., 1], knots_b[..., 1])

    # pint: (num_knots, num_knots, order + 1, order + 1)
    pint = polyint(knots_left, knots_right, prod_basis_polynomials)
    pint = torch.where(knots_left < knots_right, pint, 0.0)
    # sint: (num_knots, num_knots)
    sint = torch.sum(pint, dim=(-2, -1))
    return sint


def basis_polyval(m: torch.Tensor, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Evaluate polynomials for splines in batch mode using Horner's rule.
    # m: (num_knots, order + 1, 2)
    # p: (num_knots, order + 1, order + 1)
    # x: (batch_size, num_variables)
    #
    x_shape = list(x.shape) + [1] * (len(p.shape) - 1)
    x = x.view(x_shape)  # (batch_size, num_variables, 1, 1)
    y = polyval(p, x)    # (batch_size, num_variables, num_knots, order + 1)
    mask = (m[:, :, 0] <= x) & (x < m[:, :, 1])  # (batch_size, num_variables, num_knots, order + 1)
    y = torch.where(mask, y, 0.0)                # (batch_size, num_variables, num_knots, order + 1)
    return torch.sum(y, dim=-1)  # (batch_size, num_variables, num_knots)


def basis_polyint(m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    # Evaluate the integral of polynomials of splines.
    # m: (num_knots, order + 1, 2)
    # p: (num_knots, order + 1, order + 1)
    #
    left, right = m[:, :, 0], m[:, :, 1]
    z = polyint(left, right, p)  # (num_knots, order + 1)
    return torch.sum(z, dim=-1)  # (num_knots)


def splines_uniform_polynomial(
        order: int,
        num_knots: int,
        interval: Tuple[float, float] = (0.0, 1.0),
        clamped: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    knots = splines_uniform_knots(order, num_knots, interval=interval, clamped=clamped)
    intervals, polynomials = splines_polynomial(order, knots)
    return intervals, polynomials


def splines_uniform_knots(
        order: int,
        nun_knots: int,
        interval: Tuple[float, float] = (0.0, 1.0),
        clamped: bool = True
) -> np.ndarray:
    if nun_knots <= order:
        raise ValueError("The number of knots must be greater than the order of the polynomials")
    l, r = interval
    h = (r - l) / (nun_knots - order)
    knots = np.linspace(l - h * order, r + h * order, nun_knots + order + 1, dtype=np.float64)
    if not clamped:
        return knots
    knots = np.clip(knots, interval[0], interval[1])
    return knots


def splines_polynomial(order: int, knots: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # knots: (num_knots + order - 1)
    sidx = np.arange(len(knots) - order - 1)
    batched_spline_knots = [
        knots[sidx[i]:sidx[i] + order + 2]
        for i in range(len(sidx))
    ]
    res = list(map(build_spline_polynomial, batched_spline_knots))
    spline_intervals, spline_polynomials = zip(*res)
    spline_intervals = np.stack(spline_intervals, axis=0)      # (num_knots, order + 1, 2)
    spline_polynomials = np.stack(spline_polynomials, axis=0)  # (num_knots, order + 1, order + 1)
    return spline_intervals, spline_polynomials


def build_spline_polynomial(knots: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    knots = knots.astype(np.float64, copy=False)
    q = len(knots) - 2  # order of the polynomial

    def poly_0():
        return np.array([0.0])

    def poly_x():
        return np.array([1.0, 0.0])

    def poly_const(c: float):
        return np.array([c])

    def poly_shift(p: np.ndarray, h: float) -> np.ndarray:
        res = np.zeros([1])
        x_m_h = np.array([1, -h])
        x_m_h_p = np.ones([1])
        for i in range(len(p)):
            res = np.polyadd(res, x_m_h_p * p[-i - 1])
            x_m_h_p = np.polymul(x_m_h_p, x_m_h)
        return res

    def spline_poly_shift(i: int, k: int):
        return poly_shift(poly_x(), knots[i]) / (knots[i + k] - knots[i])

    b_prev = [
        [(knots[j], knots[j + 1], poly_const(1.0) if i == j else poly_const(0.0)) for j in range(q + 1)]
        for i in range(q + 1)
    ]

    for k in range(1, q + 1):
        b_next = []
        for i in range(q - k + 1):
            left, right = b_prev[i], b_prev[i + 1]
            if np.abs(knots[i + k] - knots[i]) < eps:
                w_left = poly_0()
            else:
                w_left = spline_poly_shift(i, k)
            if np.abs(knots[i + 1 + k] - knots[i + 1]) < eps:
                w_right = poly_0()
            else:
                w_right = np.polysub(poly_const(1.0), spline_poly_shift(i + 1, k))

            ith_b = []
            for piece in range(q + 1):
                lp, rp = left[piece], right[piece]
                poly = np.polyadd(np.polymul(lp[2], w_left), np.polymul(rp[2], w_right))
                ith_b.append((lp[0], lp[1], poly))
            b_next.append(ith_b)
        b_prev = b_next

    intervals = np.array(list(map(lambda x: (x[0], x[1]), b_prev[0])), dtype=np.float64)
    polynomials = list(map(lambda x: x[2], b_prev[0]))
    for i in range(len(polynomials)):
        len_p = len(polynomials[i])
        if len_p == q + 1:
            continue
        polynomials[i] = np.concatenate([polynomials[i], np.zeros(q + 1 - len_p)])
    polynomials = np.stack(polynomials, axis=0)
    return intervals, polynomials
