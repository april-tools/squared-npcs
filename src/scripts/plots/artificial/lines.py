import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib import pyplot as plt

from graphics.utils import setup_tueplots

parser = argparse.ArgumentParser(
    description="Plot metrics by number of parameters line graphs for the experiments on synthetic data",
)
parser.add_argument('filepath', default='', type=str, help="The processed CSV file with mean and standard deviations of metrics")
parser.add_argument('--metric', default='avg_ll', help="The metric to plot")
parser.add_argument('--legend', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--title', action='store_true', default=False)
parser.add_argument('--label', default='', help="The label of the file")

"""
python -m scripts.plots.artificial.lines results-artificial-continuous.csv --title --legend --label bsplines
python -m scripts.plots.artificial.lines results-artificial-discrete-categoricals.csv --title --legend --label categoricals
python -m scripts.plots.artificial.lines results-artificial-discrete-binomials.csv --title --legend --label binomials
"""


def format_dataset(d: str) -> str:
    if d == 'mring':
        return 'Rings'
    return d[0].upper() + d[1:]


def format_model(m: str, exp_reparam: bool = False) -> str:
    if m == 'MonotonicPC':
        return r"\textsc{MPC}"
    elif m == 'BornPC':
        if exp_reparam:
            return r"\textsc{MPC}\textsuperscript{2}"
        else:
            return r"\textsc{NPC}\textsuperscript{2}"
    assert False


def format_metric(m: str) -> str:
    if m == 'avg_ll':
        return "Average LL"
    elif m == 'bpd':
        return "Bits per dimension"
    elif m == 'ppl':
        return "Perplexity"
    assert False


def filter_dataframe(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in filter_dict.items():
        if isinstance(v, bool):
            v = float(v)
        df = df[df[k] == v]
    return df


def plot_metric_lines(df: pd.DataFrame, ax: plt.Axes):
    mean_ms = df[f'{metric}'].tolist()
    std_ms = df[f'{metric}.1'].tolist()
    ps = df['num_components'].tolist()
    mean_ms = np.array(mean_ms, dtype=np.float64)
    std_ms = np.array(std_ms, dtype=np.float64)
    ps = np.array(ps, dtype=np.int64)
    sort_indices = np.argsort(ps)
    ps = ps[sort_indices]
    mean_ms = mean_ms[sort_indices]
    std_ms = std_ms[sort_indices]
    label = f'{format_model(model_name, exp_reparam=exp_alias == "monotonic")}'
    ax.plot(ps, mean_ms, label=label, marker='o', markersize=2, linestyle='-', alpha=0.8, c=f'C{k}')
    ax.fill_between(ps, mean_ms - std_ms, mean_ms + std_ms, alpha=0.2, color=f'C{k}')


if __name__ == '__main__':
    args = parser.parse_args()
    metric = 'Best/Train/' + args.metric if args.train else 'Best/Test/' + args.metric
    df = pd.read_csv(args.filepath)

    datasets = ['mring', 'cosine', 'funnel', 'banana']
    num_rows = 1
    num_cols = len(datasets)
    setup_tueplots(num_rows, num_cols, rel_width=1.0, hw_ratio=0.8)
    fig, ax = plt.subplots(num_rows, num_cols)

    for i, dataset in enumerate(datasets):
        df_dataset = df[df['dataset'] == dataset]
        for k, (model_name, exp_alias) in enumerate(
                zip(['MonotonicPC', 'BornPC', 'BornPC'], ['', 'monotonic', 'non-monotonic'])):
            exp_reparam = exp_alias == 'monotonic'
            model_df = df_dataset[
                (df_dataset['model'] == model_name) & (df_dataset['exp_reparam'] == exp_reparam)
            ]
            plot_metric_lines(model_df, ax[i])
        ax[i].margins(0.15, 0.15)
        ax[i].grid(linestyle='--', alpha=0.25, which='both')
        #ax[i].set_xlabel(r'$K$')
        if i == 0:
            ax[i].annotate(r'$K=$', xy=(0, 0), xytext=(1, -1.5 * rcParams['xtick.major.pad']),
                           ha='right', va='top', xycoords='axes fraction', textcoords='offset points')
        if i == 0:
            ax[i].set_ylabel(format_metric(args.metric))
        if i == len(datasets) - 1 and args.legend:
            ax[i].legend(bbox_to_anchor=(1.0, 1.0))
        if args.title:
            ax[i].set_title(format_dataset(dataset))

    os.makedirs(os.path.join('figures', 'artificial-data'), exist_ok=True)
    if args.label:
        plt.savefig(os.path.join('figures', 'artificial-data', f'multi-lines-{args.metric}-{args.label}.pdf'))
    else:
        plt.savefig(os.path.join('figures', 'artificial-data', f'multi-lines-{args.metric}.pdf'))
