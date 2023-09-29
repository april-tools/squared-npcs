from typing import Optional

import numpy as np


def rotate_samples(data: np.ndarray, radia: float = np.pi * 0.25) -> np.ndarray:
    rot_data = data.copy()
    ox, oy = np.mean(data, axis=0)
    rot_data[:, 0] = ox + np.cos(radia) * (data[:, 0] - ox) - np.sin(radia) * (data[:, 1] - oy)
    rot_data[:, 1] = oy + np.sin(radia) * (data[:, 0] - ox) + np.cos(radia) * (data[:, 1] - oy)
    return rot_data


def rings_sample(num_samples: int, dim: int, sigma: float = 0.1, radia: Optional[list] = None, seed: int = 42):
    assert dim >= 2
    if radia is None:
        radia = [1, 3, 5]

    random_state = np.random.RandomState(seed)
    radia = np.asarray(radia)
    angles = random_state.rand(num_samples) * 2 * np.pi
    noise = random_state.randn(num_samples) * sigma

    weights = 2 * np.pi * radia
    weights /= np.sum(weights)

    radia_inds = random_state.choice(len(radia), num_samples, p=weights)
    radius_samples = radia[radia_inds] + noise

    xs = radius_samples * np.sin(angles)
    ys = radius_samples * np.cos(angles)
    x = np.vstack((xs, ys)).T.reshape(num_samples, 2)

    result = np.zeros((num_samples, dim))
    result[:, :2] = x
    if dim > 2:
        result[:, 2:] = random_state.randn(num_samples, dim - 2) * sigma
    return result


def single_ring_sample(num_samples: int, dim: int = 2, sigma: float = 0.26, seed: int = 42):
    return rings_sample(num_samples, dim, sigma, radia=[1], seed=seed)


def multi_rings_sample(num_samples: int, dim: int = 2, sigma: float = 0.2, seed: int = 42):
    return rings_sample(num_samples, dim, sigma, radia=[1, 3, 5], seed=seed)


def funnel_sample(num_samples: int, dim: int = 2, sigma: float = 2.0, seed: int = 42):
    def thresh(x: np.ndarray, low_lim: float = 0.0, high_lim: float = 5.0):
        return np.clip(np.exp(-x), low_lim, high_lim)
    random_state = np.random.RandomState(seed)
    data = random_state.randn(num_samples, dim)
    data[:, 0] *= sigma
    v = thresh(data[:, 0:1])
    data[:, 1:] = data[:, 1:] * np.sqrt(v)
    return data


def banana_sample(num_samples: int, dim: int = 2, sigma: float = 2.0, cf: float = 0.2, seed: int = 42):
    random_state = np.random.RandomState(seed)
    data = random_state.randn(num_samples, dim)
    data[:, 0] = sigma * data[:, 0]
    data[:, 1] = data[:, 1] + cf * (data[:, 0] ** 2 - sigma ** 2)
    if dim > 2:
        data[:, 2:] = random_state.randn(num_samples, dim - 2)
    return data


def cosine_sample(
        num_samples: int, dim: int = 2, sigma: float = 1.0,
        xlim: float = 4.0, omega: float = 2.0, alpha: float = 3.0, seed: int = 42
):
    random_state = np.random.RandomState(seed)
    x0 = random_state.uniform(-xlim, xlim, num_samples)
    x1 = alpha * np.cos(omega * x0)
    x = random_state.randn(num_samples, dim)
    x[:, 0] = x0
    x[:, 1] *= sigma
    x[:, 1] += x1
    return x


def spiral_sample(
        num_samples: int, dim: int = 2, sigma: float = 0.5, eps: float = 1.0, r_scale: float = 1.5,
        length: float = np.pi, starts: Optional[list] = None, seed: int = 42):
    if starts is None:
        starts = [0.0, 2.0 / 3, 4.0 / 3]
    starts = length * np.asarray(starts)
    nstart = len(starts)

    random_state = np.random.RandomState(seed)
    data = np.zeros((num_samples + nstart, dim))
    batch_size = np.floor_divide(num_samples + nstart, nstart)

    def branch_params(a: np.ndarray, st: float):
        n = len(a)
        a = length * (a ** (1.0 / eps)) + st
        r = (a - st) * r_scale
        m = np.zeros((n, dim))
        v = np.ones((n, dim)) * sigma
        m[:, 0] = r * np.cos(a)
        m[:, 1] = r * np.sin(a)
        v[:, :2] = (a[:, None] - st) / length * sigma + 0.1
        return m, v

    def sample_branch(n: int, st: float):
        a = random_state.uniform(0, 1, n)
        m, v = branch_params(a, st)
        return m + np.random.randn(n, dim) * v

    for si, s in enumerate(starts):
        data[si * batch_size:(si + 1) * batch_size] = sample_branch(batch_size, s)
    return data[:num_samples]
