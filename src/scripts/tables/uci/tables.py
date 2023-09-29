import sys
import os
import joblib
from copy import copy

import argparse
import numpy as np
import pandas as pd
from typing import Union, List
import json

from scripts.utils import retrieve_wandb_runs
from scripts.utils import retrieve_tboard_runs, retrieve_wandb_runs, unroll_hparams, filter_dataframe, drop_na
from datasets.loaders import CONTINUOUS_DATASETS


EXP_HPARAMS = ["num_components", "batch_size", "learning_rate"]
PARAMS_UNIT = 1_000_000
KEEP_COLS = [ "name", "model", "exp_alias", "dataset", "splines", "spline_knots", "depth", "num_replicas"]

parser = argparse.ArgumentParser(
    description="Table creation script"
)
parser.add_argument('--tboard_path', default=None, type=str, help="The Tensorboard runs path")
parser.add_argument('--wandb_path', nargs='*', default=None, type=str, help="The wandb project path user/project_name(s)")
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--metric', nargs='+', type=str, default=["Best/Test/avg_ll"], help="The metric(s) to retrieve")
parser.add_argument('--best', action='store_true', default=False)
parser.add_argument('--seed_avg', action='store_true', default=False)
parser.add_argument('--all', action='store_true', default=False)
parser.add_argument('--cache_path', type=str, default='')


def get_hparam_domains(df, match_hparams):
    hparams = dict()
    for hp in match_hparams:
        hparams[hp] = sorted(df[hp].unique())
    return hparams



def get_seed_results(exp_df, metric, search_hparams):


    assert isinstance(search_hparams, dict), "Search hparams must be list or dict"
    assert "seed" in search_hparams.keys(), "Must specify key 'seed' and domain"
    hparams_domains = copy(search_hparams)

    seeds = hparams_domains["seed"]
    del hparams_domains["seed"]
    n_seeds = len(seeds)
    print(f"Finding results for seeds {seeds}")

    # Enumerate hyperparameter name-value combinations
    hparams = unroll_hparams(hparams_domains)

    xms = list()
    for hps in hparams:
        # Filter out results for the same hparam sub-set
        hps["seed"] = seeds
        hp_df = filter_dataframe(exp_df, hps)
        
        if len(np.unique(hp_df["seed"].values)) < n_seeds:
            continue       
        else:
            # Check the seeds that are present in the results
            df_seeds = np.unique(hp_df["seed"].values).tolist()
            try:
                assert df_seeds.sort() == seeds.sort(), f"{df_seeds} != {seeds}"
            except AssertionError as e:
                print(e)
                continue

            seed_df = hp_df.groupby('seed')[metric].max().reset_index()
            assert len(seed_df) == len(seeds)
            print(f"Found results for {hps}")

            hp_df[f"{metric}_seed_avg"] = np.mean(seed_df[metric].values)
            hp_df[f"{metric}_seed_std"] = np.std(seed_df[metric].values, ddof=1)

            hp_dict = hp_df.to_dict('records')[0]
            xms.append(hp_dict)

    return pd.DataFrame(xms)  




def get_seed_averages(exp_df, metric, search_hparams):


    assert isinstance(search_hparams, dict), "Search hparams must be list or dict"
    assert "seed" in search_hparams.keys(), "Must specify key 'seed' and domain"
    hparams_domains = copy(search_hparams)

    seeds = hparams_domains["seed"]
    del hparams_domains["seed"]
    n_seeds = len(seeds)
    print(f"Averaging over seeds {seeds}")

    # Enumerate hyperparameter name-value combinations
    hparams = unroll_hparams(hparams_domains)

    xms = list()
    for hps in hparams:
        # Filter out results for the same hparam sub-set
        hps["seed"] = seeds
        hp_df = filter_dataframe(exp_df, hps)
        
        if len(np.unique(hp_df["seed"].values)) < n_seeds:
            continue       
        else:
            # Check the seeds that are present in the results
            df_seeds = np.unique(hp_df["seed"].values).tolist()
            try:
                assert df_seeds.sort() == seeds.sort(), f"{df_seeds} != {seeds}"
            except AssertionError as e:
                print(e)
                continue

            seed_df = hp_df.groupby('seed')[metric].max().reset_index()
            assert len(seed_df) == len(seeds)
            print(f"Found results for {hps}")

            hp_df[f"{metric}_seed_avg"] = np.mean(seed_df[metric].values)
            hp_df[f"{metric}_seed_std"] = np.std(seed_df[metric].values, ddof=1)

            hp_dict = hp_df.to_dict('records')[0]
            xms.append(hp_dict)

    return pd.DataFrame(xms)


def get_best_experiments(exp_df, metric, match_hparams):
    """
    Returns a table with all hyperparameter and experiment result combinations
    
    :param match_hparams: A list of hparam names or dict of hparam names and domains
    NOTE: best currently assumes MAXIMIZING given metric
    """

    # For each hyperparameter specified, find its domain
    if isinstance(match_hparams, list):
        hparams_domains = get_hparam_domains(exp_df, match_hparams)
    else:
        assert isinstance(match_hparams, dict), "Match hparams must be list or dict"
        hparams_domains = match_hparams

    # Enumerate hyperparameter name-value combinations
    hparams = unroll_hparams(hparams_domains)

    best_xms = list()
    for hps in hparams:
        # Filter out results for x and y models for the same hparam sub-set
        hp_df = filter_dataframe(exp_df, hps)
        
        if len(hp_df) == 0:
            print(f"No results for {hps}")
            continue
        elif len(hp_df) == 1:
            best_xms.append(hp_df.to_dict('records'))
        else:
            best_xms.append(hp_df.to_dict('records')[np.argmax(hp_df[metric].values)])

    return pd.DataFrame(best_xms)


def get_all_experiments(df: pd.DataFrame, match_hparams: Union[dict, list], metrics: Union[str, list], 
                 verbose: bool=True) -> pd.DataFrame:
    """
    :param df: experiment results dataframe, should have columns specified in match_hparams and metric
    :param match_hparams: either a list of hparam names to look for or a dict of hparam names and domains
    :param metrics: the metrics to collect from df (can be str if single metric, list of str if several)
    :param verbose: Print status to console
    """
    # Arg validation
    if isinstance(metrics, str):
        metrics = [metrics]

    for m in metrics:
        assert m in df.columns, f"Couldn't find metric {m} in df.columns"

    hparam_names = match_hparams if isinstance(match_hparams, list) else list(match_hparams.keys())
    
    for m in hparam_names:
        assert m in hparam_names, f"Couldn't find hparam {m} in df.columns"

    # Infer the hparam domains from the df if only names given
    if isinstance(match_hparams, list):
        hparam_domains = get_hparam_domains(df, hparam_names)
    else:
        hparam_domains = match_hparams

    # Enumerate hparam combinations and get metric result(s)
    no_results = []
    exp_results = []
    hparams = unroll_hparams(hparam_domains)
    for hps in hparams:
        hp_metric_df = filter_dataframe(df, hps)

        if len(hp_metric_df) == 0:
            no_results.append(hps)
            if verbose:
                print(f"No results for {hps}")
        elif len(hp_metric_df) == 1:
            exp_results.append(hp_metric_df.to_dict('records')[0])
        else:
            hp_metric_dics = hp_metric_df.to_dict('records')
            exp_results += hp_metric_dics

    if verbose:
        print(f"Processed total of {len(hparams)} experiment combinations, found {len(exp_results)} results.")
        print(f"No results for {len(no_results)}/{len(hparams)} experiments.")
        json.dump(no_results, open("no_result_hparams.json", "w"))

    return pd.DataFrame(exp_results)


if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(1234)

    if len(args.cache_path) != 0:
        assert os.path.isfile(args.cache_path)
        if args.verbose:
            print(f"WARNING: Loading results from CACHE {args.cache_path}")
        
        df = joblib.load(args.cache_path)

        if args.verbose:
            print(f"Loaded {len(df)} runs from CACHE {args.cache_path}")
    else:
        load_wandb = False
        load_tboard = False
        if args.wandb_path is not None:
            load_wandb = True

        if args.tboard_path is not None:
            load_tboard = True

        if not ((load_tboard or load_wandb) and (not (load_tboard and load_wandb))):
            print("Must provide one and only one of wand_path or tboard_path arguments.")
            sys.exit(-1)

        if load_tboard:
            df = retrieve_tboard_runs(args.tboard_path, args.metric)
        else:
            df = retrieve_wandb_runs(args.wandb_path, verbose=args.verbose)

        if args.verbose:
            print(f"Loaded {len(df)} runs from {args.tboard_path if load_tboard else args.wandb_path}")

        if args.verbose:
            print("Dumping results to cache file 'results_cache.jb'")
        joblib.dump(df, 'results_cache.jb')

    # Drop failed runs
    df = drop_na(df, ["Loss"])

    os.makedirs("results", exist_ok=True)
    if args.best:
        df["Best/Test/avg_ll"].fillna(df["Test/avg_ll"], inplace=True)

        df_best = drop_na(df, ["Best/Test/avg_ll"])
        df_best = get_best_experiments(df_best, "Best/Test/avg_ll", ["dataset", "splines", "exp_alias"])
        df_best.to_csv("results/best_results.csv", index=False)

    if args.seed_avg:
        df["Best/Test/avg_ll"].fillna(df["Test/avg_ll"], inplace=True)
        df_avg = drop_na(df, ["Best/Test/avg_ll"])
 
        search_hparams = {
            "num_components": [32, 64, 128, 256, 512, 1024],
            "dataset": CONTINUOUS_DATASETS,
            "splines": [True, False],
            "seed": [44, 5, 76, 52, 81],
            "model": ["BornPC"],
            "exp_alias": ["monotonic", "non-monotonic"]
        }

        df_avg_born = get_seed_averages(df_avg.copy(), "Best/Test/avg_ll", search_hparams)
        
        del search_hparams["exp_alias"]
        search_hparams["model"] = ["MonotonicPC"]
        
        df_avg_mono = get_seed_averages(df_avg, "Best/Test/avg_ll", search_hparams)

        df_avg = pd.DataFrame(df_avg_mono.to_dict('records') + df_avg_born.to_dict('records'))
        df_avg.to_csv("results/average_results.csv", index=False)
    
        
    if args.all:
        # Search for experiments with these hparams and values
        # search_hparams = {
        #     "num_components": [32, 64, 128, 256, 512, 1024],
        #     "batch_size": [512, 1024, 2048],
        #     "learning_rate": [0.01, 0.005],
        #     "dataset": CONTINUOUS_DATASETS,
        #     "splines": [True, False],
        #     "model": ["MonotonicPC", "BornPC"],
        #     "exp_alias": ["monotonic", "non-monotonic"]
        # }
        # df_all = get_all_experiments(df, search_hparams, args.metric)
        # df_all = df_all[KEEP_COLS + args.metric + ["seed"]]

        df_all = retrieve_wandb_runs(args.wandb_path)

        df_all.to_csv("results/all_results.csv", index=False)
        
    
    
