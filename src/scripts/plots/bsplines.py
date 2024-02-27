import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from graphics.utils import setup_tueplots
from splines.bsplines import basis_polyval, splines_uniform_polynomial, splines_uniform_knots

if __name__ == '__main__':
    torch.random.manual_seed(43)
    order, num_knots = 2, 7
    clamped = True
    knots = splines_uniform_knots(order=order, nun_knots=num_knots, clamped=clamped)
    print(knots)
    intervals, polynomials = splines_uniform_polynomial(order=order, num_knots=num_knots, clamped=clamped)
    intervals = torch.tensor(intervals, dtype=torch.float64)
    polynomials = torch.tensor(polynomials, dtype=torch.float64)
    if clamped:
        x = torch.linspace(0, 1 - 1e-15, steps=1024, dtype=torch.float64)
    else:
        x = torch.linspace(-0.5, 1.5 - 1e-15, steps=1024, dtype=torch.float64)
    print(intervals.shape, polynomials.shape)
    y = basis_polyval(intervals, polynomials, x)
    print(y.shape)

    num_rows = 1
    num_cols = 1
    setup_tueplots(num_rows, num_cols, rel_width=0.35, hw_ratio=0.67)
    fig, ax = plt.subplots(num_rows, num_cols)

    for i in range(y.shape[1]):
        m = y[:, i] > 7.5e-4
        ax.plot(x[m], y[m, i], linewidth=2, alpha=0.6)

    #w = 1.1 + torch.rand(y.shape[1], dtype=torch.float64)
    w = -0.2 + torch.randn(y.shape[1], dtype=torch.float64)
    z = torch.sum(y * w.unsqueeze(dim=0), dim=1)
    ax.plot(x, z, c='k', alpha=0.7, linewidth=1.5)
    ax.set_xticks(np.unique(knots.astype(np.float32)))
    ax.margins(0.1)

    os.makedirs(os.path.join('figures'), exist_ok=True)
    plt.savefig(os.path.join('figures', 'bsplines.pdf'))
