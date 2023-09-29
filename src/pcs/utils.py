from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from scipy import special

#: A random state type is either an integer seed value or a Numpy RandomState instance.
RandomState = Union[int, np.random.RandomState]


def log_binomial(n: int, k: int) -> float:
    return special.loggamma(n + 1) - special.loggamma(k + 1) - special.loggamma(n - k + 1)


def retrieve_default_dtype(numpy: bool = False) -> Union[torch.dtype, np.dtype]:
    dtype = torch.get_default_dtype()
    if not numpy:
        return dtype
    if dtype == torch.float32:
        return np.dtype(np.float32)
    if dtype == torch.float64:
        return np.dtype(np.float64)
    raise ValueError("Cannot map torch default dtype to np.dtype")


def check_random_state(random_state: Optional[RandomState] = None) -> np.random.RandomState:
    """
    Check a possible input random state and return it as a Numpy's RandomState object.

    :param random_state: The random state to check. If None a new Numpy RandomState will be returned.
                         If not None, it can be either a seed integer or a np.random.RandomState instance.
                         In the latter case, itself will be returned.
    :return: A Numpy's RandomState object.
    :raises ValueError: If the random state is not None or a seed integer or a Numpy RandomState object.
    """
    if random_state is None:
        return np.random.RandomState()
    if isinstance(random_state, int):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError("The random state must be either None, a seed integer or a Numpy RandomState object")


def num_parameters(model: nn.Module, requires_grad: bool = True) -> int:
    params = model.parameters()
    if requires_grad:
        params = filter(lambda p: p.requires_grad, params)
    return sum(p.numel() for p in params)


def safelog(x: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    if eps is None:
        eps = torch.finfo(torch.get_default_dtype()).tiny
    return torch.log(torch.clamp(x, min=eps))


@torch.no_grad()
def ohe(x: torch.Tensor, k: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if dtype is None:
        dtype = retrieve_default_dtype()
    h = torch.zeros(x.shape + (k,), dtype=dtype, device=x.device)
    h.scatter_(-1, x.unsqueeze(-1), 1)
    return h
