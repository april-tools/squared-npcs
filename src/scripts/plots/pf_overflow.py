import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from graphics.utils import setup_tueplots
from scripts.utils import setup_model, set_global_seed


def get_partition_function(num_variables: int, dtype: torch.dtype) -> Tuple[int, float]:
    torch.set_default_dtype(dtype)

    metadata = dict()
    metadata['image_shape'] = None
    metadata['num_variables'] = num_variables
    metadata['hmap'] = None
    metadata['type'] = 'continuous'
    metadata['interval'] = (-9.0, 9.0)
    metadata['domains'] = [(-9.0, 9.0) for _ in range(num_variables)]

    # Setup the model
    model = setup_model(
        'BornPC', metadata, rg_type='random',
        rg_replicas=1, rg_depth=-1, num_components=16,
        splines=False, init_method='normal'
    )
    return num_variables, torch.exp(model.log_pf()).item()


if __name__ == '__main__':
    set_global_seed(42)

    setup_tueplots(1, 1, rel_width=0.8, hw_ratio=0.8, inc_font_size=6)
    fig, ax = plt.subplots()
    ay = ax.twinx()

    ls_num_variables = [8, 16, 32, 64, 128, 256, 512]
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(ls_num_variables)
    ax.set_yticks([10.0 ** e for e in [50, 100, 150, 200]])
    ay.set_yticks([0, 120, 240, 360])
    results = dict()

    for dtype in [torch.float32, torch.float64]:
        dtype_name = 'fp32' if dtype == torch.float32 else 'fp64'
        nvs = list()
        pfs = list()
        log_pfs = list()

        for num_variables in ls_num_variables:
            num_variables, pf = get_partition_function(num_variables, dtype=dtype)
            nvs.append(num_variables)
            pfs.append(pf)
            log_pfs.append(np.log(pf))
        nvs = np.asarray(nvs)
        pfs = np.asarray(pfs)
        log_pfs = np.asarray(log_pfs)
        results[dtype_name] = {
            'vars': nvs,
            'pfs': pfs,
            'log_pfs': log_pfs
        }

    print(results)

    fp32_idx = np.argwhere(results['fp32']['pfs'] == np.inf)[0, 0]
    fp64_idx = np.argwhere(results['fp64']['pfs'] == np.inf)[0, 0]
    fp64_max = np.exp(150.0 + np.max(results['fp64']['log_pfs'][:fp64_idx]))
    ax.plot(
        results['fp32']['vars'][:fp32_idx], results['fp32']['pfs'][:fp32_idx],
        linestyle='dotted', c='C0', marker='o', markersize=6, linewidth=3, label='fp32')
    ax.hlines(
        y=fp64_max,
        xmin=results['fp32']['vars'][fp32_idx], xmax=results['fp64']['vars'][fp64_idx - 1],
        colors='C1', ls='dotted', lw=2, alpha=0.7).set_zorder(1)
    ax.hlines(
        y=fp64_max,
        xmin=results['fp64']['vars'][fp64_idx - 1], xmax=results['fp64']['vars'][fp64_idx],
        colors='C3', ls='dotted', lw=2, alpha=0.9).set_zorder(1)
    ax.plot(
        [results['fp32']['vars'][fp32_idx - 1], results['fp32']['vars'][fp32_idx]],
        [results['fp32']['pfs'][fp32_idx - 1], fp64_max],
        c='C1', ls=':', lw=2, alpha=0.7)
    ax.plot(
        [results['fp64']['vars'][fp64_idx - 1], results['fp64']['vars'][fp64_idx]],
        [results['fp64']['pfs'][fp64_idx - 1], fp64_max],
        c='C3', ls=':', lw=2, alpha=0.9)
    ax.annotate(r'$+\infty$', xy=(-0.15, 0.94), xytext=(0.0, 0.0),
                xycoords='axes fraction', textcoords='offset points')
    ax.annotate(r'$//$', xy=(-0.05, 0.825), xytext=(0.0, 0.0), rotation=-60,
                xycoords='axes fraction', textcoords='offset points')
    ax.annotate(r'$//$', xy=(0.95, 0.825), xytext=(0.0, 0.0), rotation=-60,
                xycoords='axes fraction', textcoords='offset points')
    ax.plot(
        results['fp64']['vars'][fp32_idx - 1:fp64_idx], results['fp64']['pfs'][fp32_idx - 1:fp64_idx],
        linestyle='solid', c='C0', marker='o', markersize=6, linewidth=3, label='fp64')
    ay.plot(results['fp64']['vars'][:fp64_idx], log_pfs[:fp64_idx], alpha=0.0)
    ax.scatter(
        results['fp32']['vars'][fp32_idx:fp64_idx], np.full(fp64_idx - fp32_idx, fp64_max),
        marker='X', s=150, c='C1', label=r'fp32 $+\infty$')
    ax.scatter(
        results['fp64']['vars'][fp64_idx:], np.full(len(results['fp64']['vars']) - fp64_idx, fp64_max),
        marker='X', s=150, c='C3', label=r'fp64 $+\infty$')

    ax.set_ylabel(r'$Z$')
    ay.set_ylabel(r'$\log Z$')
    ax.set_xlabel(r'Number of variables')
    ax.legend(loc='upper left', bbox_to_anchor=(0.03, 1.0), handletextpad=0.3, fontsize='x-small')
    plt.savefig(os.path.join('figures', 'partition-function-overflow.pdf'))
