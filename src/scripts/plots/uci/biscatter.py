import sys

import argparse
import itertools
import os
from typing import List, Tuple, Optional
import json

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from datasets.loaders import CONTINUOUS_DATASETS
from pcs.models import PCS_MODELS
from graphics.utils import setup_tueplots
from scripts.utils import retrieve_tboard_runs, retrieve_wandb_runs, unroll_hparams, filter_dataframe, format_model, drop_na

PALETTE = {
    "gaussian": {
        32: "#52b6bd",
        64: "#52b3ba",
        128: "#52acb3",
        256: "#519fa5",
        512: "#508589",
        1024: "#4f5152"
    },
    "splines": {
        32: "#e9635d",
        64: "#e4635c",
        128: "#db625b",
        256: "#c85f5a",
        512: "#a35956",
        1024: "#584f4f"
    }
}

dataset_title_alias = {
    "gas": "Gas",
    "power": "Power",
    "miniboone": "MiniBooNE",
    "hepmass": "Hepmass",
    "bsds300": "BSDS300"
}

parser = argparse.ArgumentParser(
    description="Bi-Scatter plot script"
)
parser.add_argument('--tboard_path', default=None, type=str, help="The Tensorboard runs path")
parser.add_argument('--wandb_path', nargs='*', default=None, type=str, help="The wandb project path user/project_name(s)")
parser.add_argument('--xmodel', choices=PCS_MODELS, required=True, help="The first model name")
parser.add_argument('--ymodel', choices=PCS_MODELS, required=True, help="The second model name")
parser.add_argument('--xalias', type=str, default='', help="The experiment alias for the first model")
parser.add_argument('--yalias', type=str, default='', help="The experiment alias for the second model")
parser.add_argument('--dataset', type=str, default='', help="The dataset to choose, omit to plot all")
parser.add_argument('--splines', action='store_true', default=False, help="Whether to get results when using splines")
parser.add_argument('--gaussians', action='store_true', default=False, help="Whether to get results with splines=False")
parser.add_argument('--match_hparams', required=True, type=str,
                    help="The list of hyperparameters to match, separated by space")
parser.add_argument('--identifier', type=str, required=True, help="An identifier of the plot")
parser.add_argument('--title', action='store_true', default=False)
parser.add_argument('--legend', action='store_true', default=False)
parser.add_argument('--ignore_diverged', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--metric', type=str, default="Best/Test/avg_ll", help="The metric to plot")
parser.add_argument('--cached_results', type=str, default='', help="CSV file with cached exp. results")
parser.add_argument('--remove_outliers', action='store_true', default=False)
"""
>For splines input units:

>Squared non-monotonic PCs vs monotonic PCs.

python -m scripts.plots.flows.biscatter <tboard-path> --xmodel "MonotonicPC" --ymodel "BornPC" \
   --yalias "non-monotonic" --dataset power --splines \
   --match-hparams "batch_size learning_rate splines spline_knots spline_order num_components" \
   --identifier mono-vs-born --title

>Squared non-monotonic PCs vs squared monotonic PCs.

python -m scripts.plots.flows.biscatter tboard-runs/uci-data-splines --xmodel "BornPC" --ymodel "BornPC" \
   --xalias "monotonic" --yalias "non-monotonic" --dataset power --splines \
   --match-hparams "batch_size learning_rate splines spline_knots spline_order num_components" \
   --identifier monoborn-vs-born

> For gaussians input units.

>Squared non-monotonic PCs vs monotonic PCs.

python -m scripts.plots.flows.biscatter <tboard-path> --xmodel "MonotonicPC" --ymodel "BornPC" \
   --yalias "non-monotonic" --dataset power \
   --match-hparams "batch_size learning_rate splines spline_knots spline_order num_components" \
   --identifier mono-vs-born --title

>Squared non-monotonic PCs vs squared monotonic PCs.

python -m scripts.plots.flows.biscatter tboard-runs/uci-data-splines --xmodel "BornPC" --ymodel "BornPC" \
   --xalias "monotonic" --yalias "non-monotonic" --dataset power \
   --match-hparams "batch_size learning_rate splines spline_knots spline_order num_components" \
   --identifier monoborn-vs-born



python src/scripts/plots/uci/biscatter.py --wandb_path ams939/born-pcs-splines ams939/born-pcs-normal --xmodel BornPC --xalias monotonic --ymodel BornPC --yalias non-monotonic --splines --gaussians --match_hparams "dataset num_components splines" --verbose --identifier "monoborn-vs-born" --title

"""


def biscatter(
        ax: plt.Axes,
        xdf: pd.DataFrame,
        ydf: pd.DataFrame,
        match_hparams: List[str],
        label: Optional[str] = None,
        jitter: float = 0.025,
        metric: str = "Best/Test/avg_ll",
        remove_outliers: bool = False,
        verbose: bool = False
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    hparams = dict()

    #xdf.to_csv("xdf.csv")
    #ydf.to_csv("ydf.csv")

    # For each hyperparameter specified, find its domain
    #for hp in match_hparams:
    #    hp_domain = sorted(xdf[hp].unique())
    #    hparams[hp] = hp_domain

    hparams = {'num_components': [32,64,128,256,512,1024],
               'splines': [True, False],
               'region_graph': ["random", "linear-vtree"]
               }

    # Enumerate hyperparameter combinations
    hparams = unroll_hparams(hparams)

    if verbose:
        print(f"Processing {len(hparams)} hparam combinations:")

    xms, yms, hms, grps = list(), list(), list(), list()
    for hps in hparams:
        # Filter out results for x and y models for the same hparam set
        xdf_filtered = filter_dataframe(xdf, hps)
        ydf_filtered = filter_dataframe(ydf, hps)
        
        xm = xdf_filtered[metric].values
        ym = ydf_filtered[metric].values
        
        if len(xm) == 0 or len(ym) == 0:
            if verbose:
                print(f"Did not find results for both models for {hps}")
            continue
        
        # xm, ym = zip(*itertools.product(xm, ym))
        # xm, ym = xm[0], ym[0]

        xm = np.amax(xm)
        ym = np.amax(ym)
        xms.append(xm)
        yms.append(ym)

        # Used for coloring points
        if 'num_components' in hps:
            hms.append(hps['num_components'])

        if 'splines' in hps:
            grps.append(hps["splines"])

    if not xms or not yms:
        if verbose:
            print("Did not find any results to plot.")
        return (0, 1), (0, 1)
    else:
        if verbose:
            print(f"Found {len(xms)} model correspondences to plot.")

    xms = np.asarray(xms)
    yms = np.asarray(yms)
    grps = np.asarray(grps)
    hms = np.asarray(hms)

    if remove_outliers:
        xo_mask = np.abs((xms - np.mean(xms)) / np.std(xms)) > 2.0
        yo_mask = np.abs((yms - np.mean(yms)) / np.std(yms)) > 2.0
        o_mask = xo_mask | yo_mask
        xms = xms[~o_mask]
        yms = yms[~o_mask]
        hms = hms[~o_mask]
        grps = grps[~o_mask]

        if verbose:
            print(f"Removed {np.sum(o_mask)} outliers")

    # Apply jitter
    xms += (np.max(xms) - np.min(xms)) * jitter * np.random.randn(*xms.shape)
    yms += (np.max(yms) - np.min(yms)) * jitter * np.random.randn(*yms.shape)

    # Plot the results
    groups = np.sort(np.unique(np.asarray(grps))).tolist()
    
    if len(groups) == 2:
        markers = {groups[0]: "o", groups[1]: "D"}
        sns.scatterplot(x=xms[grps == groups[0]], y=yms[grps == groups[0]], 
                        label=label, alpha=0.8, hue=hms[grps == groups[0]], 
                        s=20, ax=ax, 
                        palette=PALETTE["gaussian"],
                        # palette=sns.dark_palette('#27a4ad', reverse=True, as_cmap=True), 
                        style=grps[grps == groups[0]], markers=markers)

        sns.scatterplot(x=xms[grps == groups[1]], y=yms[grps == groups[1]], 
                        label=label, alpha=0.8, hue=hms[grps == groups[1]], 
                        s=20, ax=ax, 
                        palette=PALETTE["splines"],
                        # palette=sns.dark_palette('#e43d35', reverse=True, as_cmap=True), 
                        style=grps[grps == groups[1]], markers=markers)
        #69d
        #d9463e
    else:
        sns.scatterplot(x=xms, y=yms, label=label, alpha=0.5, hue=hms, s=10, ax=ax)
    
    if not args.legend:
        ax.legend([], [], frameon=False)
    
    return (np.min(xms), np.max(xms)), (np.min(yms), np.max(yms))


def plot_biscatter(
        df: pd.DataFrame,
        match_hparams: List[str],
        ax: plt.Axes,
        metric: str="Best/Test/avg_ll",
        jitter: float=0.025,
        verbose: bool = False,
        remove_outliers = False
):
    # Filter out the results for the models were interested in
    xdf = df[(df['model'] == args.xmodel) & (df['exp_alias'] == args.xalias)]
    ydf = df[(df['model'] == args.ymodel) & (df['exp_alias'] == args.yalias)]
    (xvmin, xvmax), (yvmin, yvmax) = biscatter(ax, xdf, ydf, match_hparams, metric=metric, jitter=jitter, verbose=verbose, remove_outliers=remove_outliers)

    vmin, vmax = np.min([xvmin, yvmin]), np.max([xvmin, yvmax])
    zmd = np.abs(vmax - vmin) * 0.1
    vmin, vmax = vmin - zmd, vmax + zmd
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.2, linewidth=1)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    xmodel_name = format_model(args.xmodel, exp_reparam=args.xalias == 'monotonic')
    ymodel_name = format_model(args.ymodel, exp_reparam=args.yalias == 'monotonic')

    if args.xmodel == "BornPC":
        ax.annotate(xmodel_name, xy=(0.725, 0.025), xytext=(0.0, 1.0),
                    xycoords='axes fraction', textcoords='offset points')
    else:
        ax.annotate(xmodel_name, xy=(0.800, 0.025), xytext=(0.0, 1.0),
                    xycoords='axes fraction', textcoords='offset points')
        
    ax.annotate(ymodel_name, xy=(0.015, 0.800), xytext=(1.0, 0.0),
                xycoords='axes fraction', textcoords='offset points')



def main(args):
    np.random.seed(1234)

    assert args.splines or args.gaussians, "Must provide at least one of: --splines or --gaussians"

    # If no dataset arg, process all
    if args.dataset == '':
        datasets = CONTINUOUS_DATASETS
    else:
        assert args.dataset in CONTINUOUS_DATASETS, f"Unknown dataset {args.dataset}"
        datasets = [args.dataset]

    # Load experiment results from wand, tboard or cache
    if args.cached_results == '':
        load_wandb = False
        load_tboard = False
        if args.wandb_path is not None:
            load_wandb = True

        if args.tboard_path is not None:
            load_tboard = True

        if not ((load_tboard or load_wandb) and (not (load_tboard and load_wandb))):
            print("Must provide one and only one of wand_path or tboard_path arguments.")
            sys.exit(-1)

        # df = retrieve_tboard_runs(args.tboard_path, 'Best/Test/avg_ll', ignore_diverged=args.ignore_diverged)
        if load_tboard:
            df = retrieve_tboard_runs(args.tboard_path, args.metric, ignore_diverged=args.ignore_diverged)
        else:
            df = retrieve_wandb_runs(args.wandb_path, verbose=args.verbose)

        if args.verbose:
            print(f"Found {len(df)} runs in {args.tboard_path if load_tboard else args.wandb_path}")

        # Drop failed runs
        df["Best/Test/avg_ll"].fillna(df["Test/avg_ll"], inplace=True)
        df = drop_na(df, ["Best/Test/avg_ll"])

        df.to_csv("results_cache.csv", index=False)
    else:
        assert os.path.isfile(args.cached_results), f"Invalid cache file {args.cached_results}"
        if args.verbose:
                print(f"Loading results from cache file {args.cached_results}...")
        try:
            df = pd.read_csv(args.cached_results)
        except Exception as e:
            print("Something went wrong with loading the cached results...")
            raise e
        
        if args.verbose:
            print(f"Found {len(df)} runs in {args.cached_results}")

    # Filter out Gaussians or splines if requested
    if not (args.splines and args.gaussians):
        df = filter_dataframe(df, {'splines': 1.0 if args.splines else 0.0})

        if args.verbose:
            print(f"Filtered to {len(df)} runs with dataset={args.dataset}, splines={args.splines}")
    
    plots_path = os.path.join('figures', 'uci-data')
    os.makedirs(plots_path, exist_ok=True)

    # setup_tueplots(1, 1, rel_width=1.0, hw_ratio=1.0, inc_font_size=-2)
    # fig, axs = plt.subplots(1, len(datasets))
    for idx, d in enumerate(datasets):
        if args.verbose:
            print(f"Making plot for dataset {d}...")

        # Filtering out experiments with unwanted datasets
        dataset_df = filter_dataframe(df, {
            'dataset': d
        })

        setup_tueplots(1, 1, rel_width=0.2, hw_ratio=1.0)
        plt.subplots(1, 1)
        ax = plt.gca()

        #ax = axs[idx]

        match_hparams = args.match_hparams.split()

        plot_biscatter(dataset_df, match_hparams, ax=ax, jitter=0.1, verbose=args.verbose, remove_outliers=args.remove_outliers)
        # ax.set_yticks([])
        ax.set_xticks([])
        ax.set_yticklabels(ax.get_yticks(), rotation=90)
        ax.set_aspect(1.0)

        if args.title:
            # title_kwargs = dict(ha='center', va='center', x=1.15, y=0.4, rotation=0)
            title_kwargs = dict()
            ax.set_title(dataset_title_alias[d], **title_kwargs)

        if args.splines and args.gaussians:
            alias = "splines-gaussians"
        else:
            alias = 'splines' if args.splines else 'gaussians'

        plot_file = os.path.join(plots_path, f'{args.identifier}-{d}-{alias}-biscatter.pdf')
        plt.savefig(plot_file)

        if args.verbose:
            print(f"Saved plot to {plot_file}")


def plot_legend():
    
    # Plot the legend
    K = [32, 64, 128, 256, 512, 1024]
    K.reverse()

    setup_tueplots(1, 1, rel_width=0.2, hw_ratio=1.0)
    plt.subplots(1, 1)
    ax = plt.gca()

    groups = [0.0, 1.0]
    markers = {groups[0]: "o", groups[1] : "D"}
    n = len(K)

    marker_offset = 10

    yrange = np.linspace(1, 30, num=n)
    sns.scatterplot(x=-np.ones(n)*marker_offset, y=yrange, alpha=0.8, hue=K, s=50, ax=ax, palette=sns.dark_palette('#27a4ad', reverse=True, as_cmap=True, n_colors=6), style=np.zeros(n), markers=markers)

    sns.scatterplot(x=np.ones(n)*marker_offset, y=yrange, alpha=0.8, hue=K, s=40, ax=ax, palette=sns.dark_palette('#e43d35', reverse=True, as_cmap=True, n_colors=6), style=np.ones(n), markers=markers)

    plt.rcParams['text.usetex'] = True

    ax.text(0, yrange[-1] + 1.5*((yrange[-1] - yrange[-2])), r"$K$", ha='center', va='center')
    ax.text(-marker_offset, yrange[-1] + 1.5*((yrange[-1] - yrange[-2])), "G", ha='center', va='center')
    ax.text(marker_offset, yrange[-1] + 1.5*((yrange[-1] - yrange[-2])), "S", ha='center', va='center')

    for i in range(len(K)):
        ax.text(0, yrange[i], K[i], ha='center', va='center')

    ax.set_xlim(-20, 20)
    ax.set_ylim(-5, np.amax(yrange) + 5)
    ax.legend().set_visible(False)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_aspect(1.0)

    plt.savefig("figures/legend.pdf")
    



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    plot_legend()
