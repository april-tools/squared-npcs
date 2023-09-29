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
parser.add_argument('--checkpoint-path', default='checkpoints', type=str, help="The checkpoints path")


def ring_kde() -> np.ndarray:
    splits = load_artificial_dataset('ring', num_samples=50000, dtype=np.dtype(np.float64))
    data = np.concatenate(splits, axis=0)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
    drange = np.abs(data_max - data_min)
    data_min, data_max = (data_min - drange * 0.05), (data_max + drange * 0.05)
    xlim, ylim = [(data_min[i], data_max[i]) for i in range(len(data_min))]
    return kde_samples_hmap(data, xlim=xlim, ylim=ylim, bandwidth=0.16)


def format_model_name(m: str, num_components: int) -> str:
    if m == 'MonotonicPC':
        return f"GMM ($K \! = \! {num_components}$)"
    elif m == 'BornPC':
        return f"NGMM ($K \! = \! {num_components}$)"
    return m


def load_pdf(model: str, exp_id: str) -> np.ndarray:
    filepath = os.path.join(args.checkpoint_path, 'gaussian-ring', 'ring', model, exp_id, 'pdf.npy')
    return np.load(filepath)


def plot_pdf(pdf: np.ndarray, ax: plt.Axes, vmin: Optional[float] = None, vmax: Optional[float] = None, title: Optional[str] = None, color: str = 'k'):
    xi, yi = np.mgrid[range(pdf.shape[0]), range(pdf.shape[1])]
    ax.pcolormesh(xi, yi, pdf, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    if title is not None:
        ax.set_title(title, rotation='vertical', x=-0.1, y=0.41, va='center', color=color)


if __name__ == '__main__':
    args = parser.parse_args()

    models = [
        'Ground Truth',
        'MonotonicPC',
        'MonotonicPC',
        'BornPC',
    ]

    exp_ids = [
        '',
        'RGran_R1_K2_D1_Lcp_OAdam_LR0.005_BS64_IU',
        'RGran_R1_K16_D1_Lcp_OAdam_LR0.005_BS64_IU',
        'RGran_R1_K2_D1_Lcp_OAdam_LR0.005_BS64_IN',
    ]

    truth_pdf = ring_kde()
    pdfs = [
        load_pdf(m, eid)
        for m, eid in zip(models[1:], exp_ids[1:])
    ]
    pdfs = [truth_pdf] + pdfs
    vmax = np.max(pdfs)
    vmin = 0.0

    os.makedirs(os.path.join('figures', 'gaussian-ring'), exist_ok=True)
    for idx, (p, m, eid) in enumerate(zip(pdfs, models, exp_ids)):
        setup_tueplots(1, 1, rel_width=0.2, hw_ratio=1.0)
        fig, ax = plt.subplots(1, 1)
        if eid:
            if 'PC' in m:
                num_components = int(eid.split('_')[2][1:])
            else:
                num_components = int(eid.split('_')[0][1:])
            title = f"{format_model_name(m, num_components)}"
            color = 'k' if idx == len(models) - 1 else 'k'
        else:
            title = m
            color = 'k'
        plot_pdf(p, vmin=vmin, vmax=vmax, ax=ax, title=title, color=color)
        plt.savefig(os.path.join('figures', 'gaussian-ring', f'pdfs-{idx}.png'), dpi=1200)
