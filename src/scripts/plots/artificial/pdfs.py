import argparse
import os.path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from datasets.loaders import load_artificial_dataset
from graphics.distributions import kde_samples_hmap
from graphics.utils import setup_tueplots

parser = argparse.ArgumentParser(
    description="PDFs plotter"
)
parser.add_argument('dataset', type=str, help="The artificial dataset name")
parser.add_argument('--checkpoint-path', default='checkpoints', type=str, help="The checkpoints path")

"""
python -m scripts.plots.artificial.pdfs mring && \
python -m scripts.plots.artificial.pdfs cosine && \
python -m scripts.plots.artificial.pdfs funnel && \
python -m scripts.plots.artificial.pdfs banana
"""


def artificial_kde(dataset: str, bandwidth: float = 0.2) -> np.ndarray:
    splits = load_artificial_dataset(dataset, num_samples=25000, dtype=np.dtype(np.float64))
    data = np.concatenate(splits, axis=0)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
    drange = np.abs(data_max - data_min)
    data_min, data_max = (data_min - drange * 0.05), (data_max + drange * 0.05)
    xlim, ylim = [(data_min[i], data_max[i]) for i in range(len(data_min))]
    return kde_samples_hmap(data, xlim=xlim, ylim=ylim, bandwidth=bandwidth)


def load_pdf(model: str, dataset: str, exp_alias: str, exp_id: str) -> np.ndarray:
    filepath = os.path.join(args.checkpoint_path, 'artificial-continuous', dataset, model, exp_alias, exp_id, 'pdf.npy')
    return np.load(filepath)


def plot_pdf(pdf: np.ndarray, ax: plt.Axes, vmin: Optional[float] = None, vmax: Optional[float] = None, title: Optional[str] = None):
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
        'RGran_R1_K{}_D1_Lcp_OAdam_LR0.001_BS256_SO2_SK32_IU',
        'RGran_R1_K{}_D1_Lcp_OAdam_LR0.001_BS256_SO2_SK32_RExp_IU',
        'RGran_R1_K{}_D1_Lcp_OAdam_LR0.001_BS256_SO2_SK32_IU'
    ]

    dataset_num_components = {
        'mring': 8,
        'banana': 4,
        'funnel': 4,
        'cosine': 12,
    }
    dataset_kde_bandwidths = {
        'mring': 0.05,
        'banana': 0.15,
        'funnel': 0.1,
        'cosine': 0.075,
    }

    pdfs = [
        load_pdf(m, args.dataset, eal, eid.format(dataset_num_components[args.dataset]))
        for m, eal, eid in zip(models, exp_alises, exp_ids)
    ]
    for i in range(len(pdfs)):
        p = pdfs[i]
        #p = p / np.sum(p)
        pdfs[i] = p

    truth_pdf = artificial_kde(args.dataset, bandwidth=dataset_kde_bandwidths[args.dataset])
    vmax = np.maximum(np.max(pdfs), np.max(truth_pdf))
    vmin = 0.0

    num_rows, num_cols = 1, 1
    setup_tueplots(num_rows, num_cols, rel_width=0.2, hw_ratio=1.0)
    fig, ax = plt.subplots(num_rows, num_cols)
    plot_pdf(truth_pdf, ax)
    os.makedirs(os.path.join('figures', 'artificial-data'), exist_ok=True)
    plt.savefig(os.path.join('figures', 'artificial-data', f'{args.dataset}-ground-truth.png'), dpi=1200)

    for m, eal, eid, pdf in zip(models, exp_alises, exp_ids, pdfs):
        num_rows, num_cols = 1, 1
        setup_tueplots(num_rows, num_cols, rel_width=0.2, hw_ratio=1.0)
        fig, ax = plt.subplots(num_rows, num_cols)
        plot_pdf(pdf, ax, vmin=vmin, vmax=vmax)
        plt.savefig(os.path.join('figures', 'artificial-data', f'{args.dataset}-{m}-{eal}.png'), dpi=1200)
