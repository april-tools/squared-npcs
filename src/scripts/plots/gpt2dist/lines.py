import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import rcParams

from graphics.utils import setup_tueplots
from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Plot metrics by number of parameters line graphs for the experiments on GPT2 distillation",
)
parser.add_argument('tboard_path', default='tboard-runs/gpt2-commongen-grid', type=str, help="The Tensorboard runs path")
parser.add_argument('--metric', default='avg_ll', help="The metric to plot")
parser.add_argument('--legend', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--title', action='store_true', default=False)
parser.add_argument('--median', action='store_true', default=False, help="Whether to plt min, median and max areas")

"""
python -m scripts.plots.gpt2dist.lines tboard-runs/gpt2-commongen-ihmm --median --title --train --legend & \
python -m scripts.plots.gpt2dist.lines tboard-runs/gpt2-commongen-ihmm --median --title &
"""


def format_model(m: str, exp_reparam: bool = False) -> str:
    if m == 'MonotonicPC':
        return r"$+$"
    elif m == 'BornPC':
        if exp_reparam:
            return r"$+^2$"
        else:
            return r"$\pm^2$"
    elif m == 'MonotonicHMM':
        return r"$+$"
    elif m == 'BornHMM':
        if exp_reparam:
            return r"$+^2$"
        else:
            return r"$\pm^2$"
    assert False


def format_metric(m: str) -> str:
    if m == 'avg_ll':
        return "LL"
    elif m == 'bpd':
        return "BPD"
    elif m == 'ppl':
        return "PPL"
    assert False


def filter_dataframe(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in filter_dict.items():
        if isinstance(v, bool):
            v = float(v)
        df = df[df[k] == v]
    return df


if __name__ == '__main__':
    args = parser.parse_args()
    metric = 'Best/Train/' + args.metric if args.train else 'Best/Test/' + args.metric
    df = retrieve_tboard_runs(args.tboard_path, metric, ignore_diverged=False)

    num_rows = 1
    num_cols = 1
    setup_tueplots(num_rows, num_cols, rel_width=0.3, hw_ratio=0.85)
    fig, ax = plt.subplots(num_rows, num_cols)

    markers = ['o', 'D']
    num_points = 6
    metrics = defaultdict(dict)
    model_names = ['MonotonicPC', 'BornPC']
    for k, model_name in enumerate(model_names):
        model_df = df[df['model'] == model_name]
        if model_name == 'BornPC':
            rows_to_keep = {'init_method': ['uniform', 'positive-skewed-normal', 'normal'], 'learning_rate': [5e-3, 1e-2, 5e-2]}
            model_df = model_df[model_df['init_scale'] == 0.1]
        else:
            rows_to_keep = {'init_method': ['uniform', 'dirichlet', 'log-normal'], 'learning_rate': [5e-3, 1e-2, 5e-2]}
            model_df = model_df[model_df['init_scale'] == 1.0]
        if rows_to_keep is not None:
            for r, vs in rows_to_keep.items():
                model_df = model_df[model_df[r].isin(vs)]
        model_df.to_csv(f'{model_name}-gpt2commongen-results.csv', index=None)
        group_model_df = model_df.groupby(by=['init_method', 'learning_rate'])
        should_label = True
        metrics[model_name] = defaultdict(list)
        for j, hparam_df in group_model_df:
            ms, ps = hparam_df[metric].tolist(), hparam_df['num_components'].tolist()
            if len(np.unique(ms)) < num_points or len(np.unique(ps)) < num_points:
                continue
            ms = np.array(ms, dtype=np.float64)
            ps = np.array(ps, dtype=np.int64)
            sort_indices = np.argsort(ps)
            ps = ps[sort_indices]
            ms = ms[sort_indices]
            ps = ps[:num_points]
            ms = ms[:num_points]
            for p, m in zip(ps.tolist(), ms.tolist()):
                metrics[model_name][p].append(m)
            if not args.median:
                should_label = False
                label = f'{format_model(model_name)}' if should_label else None
                ax.plot(ps, ms, label=label, marker=markers[k], markersize=3, linewidth=1.5, linestyle='-', alpha=0.2, c=f'C{k}')
        if args.median:
            label = f'{format_model(model_name)}' if should_label else None
            median_metrics = sorted(list(map(
                lambda x: (x[0], np.median(x[1])), metrics[model_name].items())),
                key=lambda x: x[0])
            bot_metrics = sorted(list(map(
                lambda x: (x[0], np.quantile(x[1], q=0.05)), metrics[model_name].items())),
                key=lambda x: x[0])
            top_metrics = sorted(list(map(
                lambda x: (x[0], np.quantile(x[1], q=0.95)), metrics[model_name].items())),
                key=lambda x: x[0])
            median_ps, median_ms = zip(*median_metrics)
            _, bot_ms = zip(*bot_metrics)
            _, top_ms = zip(*top_metrics)
            ax.plot(median_ps, median_ms, label=label, marker=markers[k], markersize=3, linewidth=1.5, linestyle='-', alpha=0.8, c=f'C{k}')
            ax.fill_between(median_ps, bot_ms, top_ms, alpha=0.2, color=f'C{k}')

    assert len(model_names) == 2
    model_a, model_b = model_names
    spvalues = defaultdict(lambda: defaultdict(dict))
    for ts in ['mannwithneyu', 'ttest']:
        for al in ['greater']:
            for k in sorted(metrics[model_a].keys() & metrics[model_b].keys()):
                lls_a = metrics[model_a][k]
                lls_b = metrics[model_b][k]
                if ts == 'mannwithneyu':
                    s, p = stats.mannwhitneyu(lls_b, lls_a, method='exact', alternative=al)
                elif ts == 'ttest':
                    s, p = stats.ttest_ind(lls_b, lls_a, alternative=al)
                else:
                    assert False, "Should not happen :("
                spvalues[ts][al][k] = (round(s, 3), round(p, 4))
    print(spvalues)

    #if args.train:
    #    gpt2_average_log_likelihood = -52.01387770076976
    #    ax.axhline(y=gpt2_average_log_likelihood, color='k', linewidth=2, linestyle='dotted', label='GPT2')
    ax.margins(0.15, 0.15)
    ax.grid(linestyle='--', alpha=0.3, which='both')
    #ax.set_xlabel(r'$K$')
    #ax.annotate(r'$K$', xy=(1, 0), xytext=(1, -1.5 * rcParams['xtick.major.pad']),
    #            ha='right', va='top', xycoords='axes fraction', textcoords='offset points')
    if args.train:
        ax.annotate(r'$K=$', xy=(0, 0), xytext=(1, -1.5 * rcParams['xtick.major.pad']),
                    ha='right', va='top', xycoords='axes fraction', textcoords='offset points')
    #ax.set_ylabel(format_metric(args.metric))
    ax.annotate(f'{format_metric(args.metric)}', xy=(0, 1.05), xytext=(-1.5 * rcParams['ytick.major.pad'], 1),
                ha='right', va='bottom', xycoords='axes fraction', textcoords='offset points')
    ax.set_xscale('log')
    if args.train:
        #ax.set_yticks([-80.0, -75.0, -70.0, -65.0])
        ax.set_yticks([-80.0, -75.0, -70.0])
    else:
        #ax.set_yticks([-90.0, -85.0, -80.0, -75.0])
        ax.set_yticks([-85.0, -80.0, -75.0])
    if args.legend:
        ax.legend()
    if args.title:
        ax.set_title('Training data' if args.train else 'Test data')

    os.makedirs(os.path.join('figures', 'gpt2dist'), exist_ok=True)
    plt.savefig(os.path.join('figures', 'gpt2dist', f'multi-lines-{args.metric}-{"train" if args.train else "test"}.pdf'))
