import argparse
import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from graphics.utils import setup_tueplots
from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Plot metrics by number of parameters line graphs",
)
parser.add_argument('tboard_path', default='tboard-runs', type=str, help="The Tensorboard runs path")
parser.add_argument('--metric', default='avg_ll', help="The metric to plot")
parser.add_argument('--datasets', type=str, default='power;gas;hepmass;miniboone', help="Dataset names")
parser.add_argument('--models', default='MonotonicPC;BornPC', help="The models")
parser.add_argument('--exp-aliases', default='', help="The experiment aliases")
parser.add_argument('--filename', type=str, required=True, help="The name of the file where to save the plot")
parser.add_argument('--legend', action='store_true', default=False)
parser.add_argument('--title', action='store_true', default=False)


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


if __name__ == '__main__':
    args = parser.parse_args()
    metric = 'Best/Test/' + args.metric
    df = retrieve_tboard_runs(args.tboard_path, metric)

    datasets = args.datasets.split(';')
    hparam_settings = [
        #{'splines': [False], 'learning_rate': [0.001, 0.005]},
        #{'splines': [True], 'learning_rate': [0.001, 0.005], 'spline_knots': [128, 256, 512]}
        {}
    ]
    num_rows = len(hparam_settings)
    num_cols = len(datasets)
    models = args.models.split(';')

    if args.exp_aliases:
        exp_aliases = args.exp_aliases.split(';')
        if len(exp_aliases) == 1:
            exp_aliases = exp_aliases * len(models)
    else:
        exp_aliases = [''] * len(models)

    setup_tueplots(num_rows, num_cols, rel_width=1.0, hw_ratio=1.0)
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=False)

    for i, hparams in enumerate(hparam_settings):
        for j, dataset in enumerate(datasets):
            df_dataset = df[df['dataset'] == dataset]
            for k, (model, alias) in enumerate(zip(models, exp_aliases)):
                df_model = df_dataset[df_dataset['model'] == model]
                df_model = df_model[df_model['exp_alias'] == alias]
                hparams_dicts = [dict(zip(hparams.keys(), values)) for values in itertools.product(*hparams.values())]
                for hps in hparams_dicts:
                    hps['learning_rate'] = 0.01
                    df_hp = filter_dataframe(df_model, hps)
                    if len(df_hp) == 0:
                        continue

                    ms, ps = df_hp[metric].tolist(), df_hp['num_params'].tolist()
                    ms = np.array(ms, dtype=np.float64)
                    ps = np.array(ps, dtype=np.int64)
                    sort_indices = np.argsort(ps)
                    ps = ps[sort_indices]
                    ms = ms[sort_indices]

                    label = model
                    if alias:
                        label = f'{label} ({"".join(map(lambda a: a[0].upper(), alias.split("-")))})'
                    ax[i, j].plot(ps, ms, label=label, marker='o', markersize=2, alpha=0.333, c=f'C{k}')

            #if j == 0:
            #    ax[i, j].set_ylabel(format_metric(args.metric))
            #if i == len(hparam_settings) - 1:
            #    ax[i, j].set_xlabel(r'\# of parameters')
            if args.legend and i == 0 and j == len(datasets) - 1:
                ax[i, j].legend(loc='upper right', fontsize=4, framealpha=0.2)
            #ax[i, j].set_aspect(1.0)
            if args.title and i == 0:
                ax[i, j].set_title(dataset.upper())
            ax[i, j].set_xscale('log')
            ax[i, j].grid(linestyle='--', alpha=0.3, linewidth=.5)
            ax[i, j].margins(x=0.1, y=0.1)

    plt.savefig(os.path.join('figures', f'{args.filename}-mplines.pdf'))
