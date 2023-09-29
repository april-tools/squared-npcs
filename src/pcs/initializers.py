import torch
import numpy as np
from torch import nn
from torch import distributions

INIT_METHODS = [
    'uniform',
    'normal',
    'log-normal',
    'negative-skewed-normal',
    'negative-skewed-log-normal',
    'positive-skewed-normal',
    'positive-skewed-log-normal',
    'fw-normal',
    'fw-log-normal',
    'gamma',
    'dirichlet',
    'xavier-uniform',
    'xavier-normal'
]


def init_params_(tensor: torch.Tensor, method: str = 'normal', init_loc: float = 0.0, init_scale: float = 1.0):
    if init_scale < 1e-9:
        raise ValueError("Initialization scale is too small")
    if method == 'uniform':
        init_loc = init_loc + torch.finfo(torch.get_default_dtype()).tiny
        nn.init.uniform_(tensor, a=init_loc, b=init_loc + init_scale)
    elif method == 'normal':
        nn.init.normal_(tensor, init_loc, init_scale)
    elif method == 'log-normal':
        nn.init.normal_(tensor, init_loc, init_scale)
        tensor.exp_()
    elif method == 'negative-skewed-normal':
        nn.init.normal_(tensor, init_loc - init_scale, init_scale)
    elif method == 'negative-skewed-log-normal':
        nn.init.normal_(tensor, init_loc - init_scale, init_scale)
    elif method == 'positive-skewed-normal':
        nn.init.normal_(tensor, init_loc + init_scale, init_scale)
    elif method == 'positive-skewed-log-normal':
        nn.init.normal_(tensor, init_loc + init_scale, init_scale)
    elif method == 'fw-normal':
        init_loc = np.log(tensor.shape[-1]) / 3.0 + 0.5 * (init_scale ** 2)
        t = torch.randn(*tensor.shape) * init_scale - init_loc
        tensor.copy_(t)
    elif method == 'fw-log-normal':
        init_loc = np.log(tensor.shape[-1]) / 3.0 + 0.5 * (init_scale ** 2)
        t = torch.exp(torch.randn(*tensor.shape) * init_scale - init_loc)
        tensor.copy_(t)
    elif method == 'gamma':
        gamma_(tensor, init_loc + 2.0, init_scale)
    elif method == 'dirichlet':
        dirichlet_(tensor, alpha=1.0 / init_scale, log_space=False)
    elif method == 'xavier-uniform':
        fan_in, fan_out = tensor.shape[-1], tensor.shape[-2]
        std = np.sqrt(2.0 / float(fan_in + fan_out))
        a = np.sqrt(3.0) * std
        return nn.init.uniform_(tensor, -a, a)
    elif method == 'xavier-normal':
        fan_in, fan_out = tensor.shape[-1], tensor.shape[-2]
        std = np.sqrt(2.0 / float(fan_in + fan_out))
        return nn.init.normal_(tensor, 0.0, std)
    else:
        raise NotImplementedError(f"Unknown initialization method called {method}")


def gamma_(tensor: torch.Tensor, alpha: float = 1.0, beta: float = 1.0):
    with torch.no_grad():
        samples = distributions.Gamma(
            torch.full_like(tensor, fill_value=alpha),
            torch.full_like(tensor, fill_value=beta)
        ).sample()
        tensor.copy_(samples)


def dirichlet_(tensor: torch.Tensor, alpha: float = 1.0, log_space: bool = False, dim: int = -1):
    """
    Initialize a tensor using the symmetric Dirichlet distribution.

    :param tensor: The tensor to initialize.
    :param alpha: The concentration parameter.
    :param log_space: Whether to initialize the tensor in the logarithmic space.
    :param dim: The dimension over which to sample.
    """
    shape = tensor.shape
    if len(shape) == 0:
        raise ValueError("Singleton tensors are not valid")
    with torch.no_grad():
        idx = (len(shape) + dim) % len(shape)
        concentration = torch.full([shape[idx]], fill_value=alpha)
        dirichlet = distributions.Dirichlet(concentration)
        samples = dirichlet.sample(torch.Size([d for i, d in enumerate(shape) if i != idx]))
        if log_space:
            samples = torch.log(samples)
        tensor.copy_(torch.transpose(samples, idx, -1))
