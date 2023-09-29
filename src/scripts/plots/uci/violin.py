import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from graphics.utils import setup_tueplots
from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Plot metrics as horizontal violin plots",
)
parser.add_argument('tboard_path', default='tboard-runs', type=str, help="The Tensorboard runs path")
parser.add_argument('--metric', default='avg_ll', help="The metric to plot")
parser.add_argument('--models', default='MonotonicPC;BornPC', help="The models")
parser.add_argument('--exp-aliases', default='', help="The experiment aliases")
parser.add_argument('--identifier', type=str, required=True, help="An identifier of the plot")
parser.add_argument('--title', action='store_true', default=False)
"""
python -m scripts.plots.flows.violin tboard-runs/flow-data \
  --models "MonotonicPC;BornPC" --identifier mono-vs-born --title
"""


def format_metric(m: str) -> str:
    if m == 'avg_ll':
        return "Average LL"
    elif m == 'bpd':
        return "Bits per dimension"
    elif m == 'ppl':
        return "Perplexity"
    assert False


def format_model(m: str, exp_reparam: bool = False) -> str:
    if m == 'MonotonicPC':
        return r"$+$"
    elif m == 'BornPC':
        if exp_reparam:
            return r"$+^2$"
        else:
            return r"$\pm^2$"
    assert False


def filter_dataframe(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in filter_dict.items():
        if isinstance(v, bool):
            v = float(v)
        df = df[df[k] == v]
    return df


def plot_violin(
        df: pd.DataFrame,
        dataset: str,
        models: List[str],
        exp_aliases: List[str],
        ax: plt.Axes,
        split: bool = False
):
    y_violin_data, x_violin_data, h_violin_data = list(), list(), list()
    df = df[df['dataset'] == dataset]
    for k, (model, alias) in enumerate(zip(models, exp_aliases)):
        df_model = df[(df['model'] == model) & (df['exp_alias'] == alias)]
        hs = list('Gaussians' if sf == 0 else 'Splines' for sf in df_model['splines'].tolist())
        ms = df_model[metric].tolist()
        label = format_model(model, exp_reparam=alias == 'monotonic')
        if alias:
            label = f'{label} ({"".join(map(lambda a: a[0].upper(), alias.split("-")))})'
        y_violin_data.extend([label] * len(ms))
        x_violin_data.extend(ms)
        h_violin_data.extend(hs)
    if split:
        sns.violinplot(x=x_violin_data, y=y_violin_data, hue=h_violin_data, split=True, linewidth=1, ax=ax)
    else:
        sns.violinplot(x=x_violin_data, y=y_violin_data, linewidth=1, ax=ax)


if __name__ == '__main__':
    args = parser.parse_args()
    metric = 'Best/Test/' + args.metric
    df = retrieve_tboard_runs(args.tboard_path, metric)
    models = args.models.split(';')

    if args.exp_aliases:
        exp_aliases = args.exp_aliases.split(';')
        if len(exp_aliases) == 1:
            exp_aliases = exp_aliases * len(models)
    else:
        exp_aliases = [''] * len(models)

    setup_tueplots(2, 2, rel_width=1.0, hw_ratio=0.667)
    fig, ax = plt.subplots(2, 2)

    plot_violin(df, 'power', models, exp_aliases, ax[0, 0])
    plot_violin(df, 'gas', models, exp_aliases, ax[0, 1])
    plot_violin(df, 'hepmass', models, exp_aliases, ax[1, 0])
    plot_violin(df, 'miniboone', models, exp_aliases, ax[1, 1])

    ax[0, 1].set_yticks([])
    ax[1, 1].set_yticks([])
    ax[1, 0].set_xlabel(format_metric(args.metric))
    ax[1, 1].set_xlabel(format_metric(args.metric))

    if args.title:
        title_kwargs = dict(ha='right', va='center', x=1.05, y=0.5, rotation=270)
        ax[0, 0].set_title('POWER', **title_kwargs)
        ax[0, 1].set_title('GAS', **title_kwargs)
        ax[1, 0].set_title('HEPMASS', **title_kwargs)
        ax[1, 1].set_title('MINIBOONE', **title_kwargs)

    os.makedirs(os.path.join('figures', 'flow-data'), exist_ok=True)
    plt.savefig(os.path.join('figures', 'flow-data', f'{args.identifier}-violin.pdf'))
