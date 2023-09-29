from typing import Union, Optional, Tuple

import numpy as np
import torch
from torchvision.utils import make_grid

from pcs.models import PC
from pcs.sampling import inverse_transform_sample


def sample_model_images(
    model: PC,
    shape: Tuple[int, int, int],
    vdomain: int = 256,
    num_samples: int = 1,
    nrow: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None
) -> np.ndarray:
    model.eval()
    samples = list()
    for _ in range(num_samples):
        samples.append(inverse_transform_sample(model, vdomain=vdomain, num_samples=1, device=device))
    samples = torch.stack(samples) * np.reciprocal(vdomain - 1.0)
    samples = samples.view(-1, *shape)
    grid = make_grid(samples, nrow=nrow, padding=0)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    return ndarr
