import argparse
import os.path
from typing import Optional

import matplotlib as mpl
import numpy as np
from scipy import special
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from datasets.loaders import load_artificial_dataset
from graphics.distributions import kde_samples_hmap
from graphics.utils import setup_tueplots
from pcs.models import TensorizedPC, PC, MonotonicPC
from scripts.utils import setup_model, setup_data_loaders

parser = argparse.ArgumentParser(
    description="PDFs and ellipses plotter"
)
parser.add_argument('--checkpoint-path', default='checkpoints', type=str, help="The checkpoints path")
parser.add_argument('--show-ellipses', default=False, action='store_true',
                    help="Whether to show the Gaussian components as ellipses")
parser.add_argument('--title', default=False, action='store_true', help="Whether to show a title")


def format_model_name(m: str, num_components: int) -> str:
    if m == 'MonotonicPC':
        return f"GMM ($K \!\! = \!\! {num_components}$)"
    elif m == 'BornPC':
        return f"NGMM ($K \!\! = \!\! {num_components}$)"
    return m


def load_mixture(
        model_name: str,
        exp_id_fmt: str,
        num_components: int,
        learning_rate: float = 5e-3,
        batch_size: int = 64
) -> TensorizedPC:
    metadata, _ = setup_data_loaders('ring', 'datasets', 1)
    model: TensorizedPC = setup_model(model_name, metadata, num_components=num_components)
    exp_id = exp_id_fmt.format(num_components, learning_rate, batch_size)
    filepath = os.path.join(args.checkpoint_path, 'ring', model_name, exp_id, 'model.pt')
    state_dict = torch.load(filepath, map_location='cpu')
    model.load_state_dict(state_dict['weights'])
    return model


def load_pdf(
        model: str,
        exp_id_fmt: str,
        num_components,
        learning_rate: float = 5e-3,
        batch_size: int = 64
) -> np.ndarray:
    exp_id = exp_id_fmt.format(num_components, learning_rate, batch_size)
    filepath = os.path.join(args.checkpoint_path, 'ring', model, exp_id, 'distbest.npy')
    return np.load(filepath)


def plot_mixture_ellipses(mixture: TensorizedPC, ax: plt.Axes):
    mus = mixture.input_layer.mu[0, :, 0, :].detach().numpy()
    covs = np.exp(2 * mixture.input_layer.log_sigma[0, :, 0, :].detach().numpy())
    num_components = mus.shape[-1]
    mix_weights = mixture.layers[-1].weight[0, 0].detach().numpy()
    if isinstance(mixture, MonotonicPC):
        mix_weights = special.softmax(mix_weights)
        mix_weights = mix_weights / np.max(mix_weights)
    else:
        mix_weights = -mix_weights / np.max(np.abs(mix_weights))
    for i in range(num_components):
        if np.abs(mix_weights[i]) < 0.1:
            continue
        mu = mus[:, i]
        cov = np.diag(covs[:, i])
        v, w = np.linalg.eigh(cov)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mu, v[0], v[1], linewidth=0.7, fill=False)
        ell_dot = mpl.patches.Circle(mu, radius=0.03, fill=True)
        ell.set_color('red')
        if isinstance(mixture, MonotonicPC):
            #ell.set_alpha(mix_weights[i])
            #ell_dot.set_alpha(0.5 * mix_weights[i])
            ell_dot.set_color('red')
            ell.set_alpha(0.775)
            ell_dot.set_alpha(0.775)
        else:
            if mix_weights[i] <= 0.0:
                #ell.set_alpha(min(1.0, 3 * np.abs(mix_weights[i])))
                ell.set_linestyle('dotted')
                #ell_dot.set_alpha(0.5 * np.abs(mix_weights[i]))
                #%ell_dot.set_color('red')
            else:
                #ell.set_alpha(mix_weights[i])
                #ell_dot.set_alpha(0.5 * mix_weights[i])
                ell_dot.set_color('red')
            ell.set_alpha(0.85)
            ell_dot.set_alpha(0.85)
        ax.add_artist(ell)
        ax.add_artist(ell_dot)


def plot_pdf(
        pdf: np.ndarray,
        metadata: dict,
        ax: plt.Axes, vmin:
        Optional[float] = None,
        vmax: Optional[float] = None
):
    pdf = pdf[8:-8, 8:-8]

    x_lim = metadata['domains'][0]
    y_lim = metadata['domains'][1]
    x_lim = (x_lim[0] * np.sqrt(2.0), x_lim[1] * np.sqrt(2.0))
    y_lim = (y_lim[0] * np.sqrt(2.0), y_lim[1] * np.sqrt(2.0))

    x_lim = (min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1]))
    y_lim = (min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1]))

    xi, yi = np.mgrid[range(pdf.shape[0]), range(pdf.shape[1])]
    xi = (xi + 0.5) / pdf.shape[0]
    yi = (yi + 0.5) / pdf.shape[1]
    xi = xi * (x_lim[1] - x_lim[0]) + x_lim[0]
    yi = yi * (y_lim[1] - y_lim[0]) + y_lim[0]
    ax.pcolormesh(xi, yi, pdf, vmin=vmin, vmax=vmax)


if __name__ == '__main__':
    args = parser.parse_args()

    models = [
        'MonotonicPC',
        'MonotonicPC',
        'BornPC'
    ]

    num_components = [2, 16, 2]
    learning_rates = [5e-3, 5e-3, 1e-3]

    exp_id_formats = [
        'RGran_R1_K{}_D1_Lcp_OAdam_LR{}_BS{}_IU',
        'RGran_R1_K{}_D1_Lcp_OAdam_LR{}_BS{}_IU',
        'RGran_R1_K{}_D1_Lcp_OAdam_LR{}_BS{}_IN'
    ]

    truth_pdf = np.load(
        os.path.join(args.checkpoint_path, 'ring', models[0],
                     exp_id_formats[0].format(num_components[0], learning_rates[0], 64), 'gt.npy')
    )
    # truth_pdf = ring_kde()

    mixtures = [
        load_mixture(m, eif, nc, lr)
        for m, eif, nc, lr in zip(models, exp_id_formats, num_components, learning_rates)
    ]

    pdfs = [
        load_pdf(m, eif, nc, lr)
        for m, eif, nc, lr in zip(models, exp_id_formats, num_components, learning_rates)
    ]
    vmax = np.max([truth_pdf] + pdfs)
    vmin = 0.0

    metadata, _ = setup_data_loaders('ring', 'datasets', 1, num_samples=10000)

    os.makedirs(os.path.join('figures', 'gaussian-ring'), exist_ok=True)
    data_pdfs = [(None, truth_pdf, 'Ground Truth', -1)] + list(zip(mixtures, pdfs, models, num_components))
    for idx, (p, pdf, m, nc) in enumerate(data_pdfs):
        setup_tueplots(1, 1, rel_width=0.2, hw_ratio=1.0)
        fig, ax = plt.subplots(1, 1)
        if args.title:
            title = f"{format_model_name(m, nc)}" if p is not None else m
        else:
            title = None

        plot_pdf(pdf, metadata, ax=ax, vmin=vmin, vmax=vmax)
        if p is not None and args.show_ellipses:
            plot_mixture_ellipses(p, ax=ax)

        x_lim = metadata['domains'][0]
        y_lim = metadata['domains'][1]
        x_lim = (x_lim[0] * np.sqrt(2.0), x_lim[1] * np.sqrt(2.0))
        y_lim = (y_lim[0] * np.sqrt(2.0), y_lim[1] * np.sqrt(2.0))
        #lims = (min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1]))

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.0)
        if args.title:
            ax.set_title(title, rotation='vertical', x=-0.1, y=0.41, va='center')

        filename = f'pdfs-ellipses-{idx}.png' if args.show_ellipses else f'pdfs-{idx}.png'
        plt.savefig(os.path.join('figures', 'gaussian-ring', filename), dpi=1200)
