import argparse
import os.path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from datasets.loaders import load_artificial_dataset
from graphics.distributions import kde_samples_hmap, plot_bivariate_discrete_samples_hmap, discrete_samples_hmap
from graphics.utils import setup_tueplots

parser = argparse.ArgumentParser(
    description="PDFs plotter"
)
parser.add_argument('dataset', type=str, help="The artificial dataset name")
parser.add_argument('--discretize-bins', required=True, type=int, help="The number of bins for discretization")
parser.add_argument('--checkpoint-path', default='checkpoints', type=str, help="The checkpoints path")

"""
python -m scripts.plots.artificial.pmfs mring --discretize-bins 32 && \
    python -m scripts.plots.artificial.pmfs cosine --discretize-bins 32 && \
    python -m scripts.plots.artificial.pmfs funnel --discretize-bins 32 && \
    python -m scripts.plots.artificial.pmfs banana --discretize-bins 32
"""


def artificial_pmf(dataset: str, discretize_bins: int) -> np.ndarray:
    splits = load_artificial_dataset(
        dataset, num_samples=25000, dtype=np.dtype(np.float64),
        discretize=True, discretize_bins=discretize_bins)
    data = np.concatenate(splits, axis=0)
    return discrete_samples_hmap(data, xlim=(0, discretize_bins - 1), ylim=(0, discretize_bins - 1))


def load_pmf(model: str, dataset: str, exp_alias: str, exp_id: str, binomials: bool = False) -> np.ndarray:
    suffix = 'binomials' if binomials else 'categoricals'
    filepath = os.path.join(args.checkpoint_path, f'artificial-discrete-{suffix}',
                            dataset, model, exp_alias, exp_id, 'pmf.npy')
    return np.load(filepath)


def plot_pmf(pdf: np.ndarray, ax: plt.Axes, vmin: Optional[float] = None, vmax: Optional[float] = None, title: Optional[str] = None):
    xi, yi = np.mgrid[range(pdf.shape[0]), range(pdf.shape[1])]
    ax.pcolormesh(xi, yi, pdf, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    if title is not None:
        ax.set_title(title, y=-0.275)


if __name__ == '__main__':
    args = parser.parse_args()

    models = [
        'MonotonicPC',
        'BornPC',
        'BornPC'
    ]

    exp_alises = [
        '',
        'monotonic',
        'non-monotonic'
    ]

    exp_ids = [
        'RGran_R1_K{}_D1_Lcp_OAdam_LR0.001_BS256_IU',
        'RGran_R1_K{}_D1_Lcp_OAdam_LR0.001_BS256_RExp_IU',
        'RGran_R1_K{}_D1_Lcp_OAdam_LR0.001_BS256_IU'
    ]

    dataset_num_components = {
        'categoricals': {
            'mring': 8,
            'banana': 4,
            'funnel': 4,
            'cosine': 12,
        },
        'binomials': {
            'mring': 96,
            'banana': 32,
            'funnel': 64,
            'cosine': 64
        }
    }

    truth_pmf = artificial_pmf(args.dataset, discretize_bins=args.discretize_bins)
    pmfs_categoricals = [
        load_pmf(
            m, args.dataset, f'{eal}-b{args.discretize_bins}' if eal else f'b{args.discretize_bins}',
            eid.format(dataset_num_components['categoricals'][args.dataset]), binomials=False)
        for m, eal, eid in zip(models, exp_alises, exp_ids)
    ]
    pmfs_binomials = [
        load_pmf(
            m, args.dataset, f'{eal}-b{args.discretize_bins}' if eal else f'b{args.discretize_bins}',
            eid.format(dataset_num_components['binomials'][args.dataset]), binomials=True)
        for m, eal, eid in zip(models, exp_alises, exp_ids)
    ]
    vmax = np.maximum(np.maximum(np.max(truth_pmf), np.max(pmfs_categoricals)), np.max(pmfs_binomials))
    vmin = 0.0
    os.makedirs(os.path.join('figures', 'artificial-data'), exist_ok=True)

    num_rows, num_cols = 1, 1
    setup_tueplots(num_rows, num_cols, rel_width=0.2, hw_ratio=1.0)
    fig, ax = plt.subplots(num_rows, num_cols)
    plot_pmf(truth_pmf, vmax=vmax, ax=ax)
    plt.savefig(os.path.join('figures', 'artificial-data', f'{args.dataset}-ground-truth-pmf.png'), dpi=1200)

    for idx, (m, eal, eid, pmf) in enumerate(zip(models, exp_alises, exp_ids, pmfs_categoricals)):
        num_rows, num_cols = 1, 1
        setup_tueplots(num_rows, num_cols, rel_width=0.2, hw_ratio=1.0)
        fig, ax = plt.subplots(num_rows, num_cols)
        plot_pmf(pmf, ax, vmin=vmin, vmax=vmax)
        plt.savefig(os.path.join('figures', 'artificial-data', f'{args.dataset}-{m}-{eal}-categoricals.png'), dpi=1200)

    for idx, (m, eal, eid, pmf) in enumerate(zip(models, exp_alises, exp_ids, pmfs_binomials)):
        num_rows, num_cols = 1, 1
        setup_tueplots(num_rows, num_cols, rel_width=0.2, hw_ratio=1.0)
        fig, ax = plt.subplots(num_rows, num_cols)
        plot_pmf(pmf, ax, vmin=vmin, vmax=vmax)
        plt.savefig(os.path.join('figures', 'artificial-data', f'{args.dataset}-{m}-{eal}-binomials.png'), dpi=1200)
