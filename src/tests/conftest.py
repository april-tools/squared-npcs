import pytest

import torch

from scripts.utils import set_global_seed


@pytest.fixture(autouse=True)
def setup_globals():
    set_global_seed(42)
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float64)
