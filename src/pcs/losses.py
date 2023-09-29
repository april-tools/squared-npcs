import torch


def nll(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(x)


def setup_loss(loss: str):
    if loss == 'nll':
        return nll
    raise ValueError(f"Unknown loss function called {loss}")
