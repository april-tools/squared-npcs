import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Best hyperparameters"
)
parser.add_argument('--tboard-hparams', default='', type=str, help="The Tensorboard HPARAMS CSV file")
parser.add_argument('--tboard-path', default='', type=str, help="The Tensorboard runs path")
parser.add_argument('best_filepath', type=str, help="The filepath where to save the best hyperparameters results")
parser.add_argument(
    '--cmd-filepath', type=str, default='',
    help="The file where to save the list of commands to replicate the results")
parser.add_argument(
    '--best-lower', action='store_true',
    help="Whether to take as best experiments those with lower validation metrics")
parser.add_argument('--metric', type=str, default='avg_ll',
                    choices=['avg_ll', 'bpd', 'ppl'], help="The validation metric")
parser.add_argument('--digits', type=int, default=3, help="Maximum number of digits after comma")
parser.add_argument('--group-keys', type=str, default='dataset model', help="The keys of the dataframe to group")
parser.add_argument(
    '--average-reps', action='store_true', default=False,
    help="Whether to average multiple repetitions and run statistical tests"
)
"""
python -m scripts.results --tboard-hparams hparams-artificial-continuous.csv results-artificial-continuous.csv \
    --average-reps --group-keys "dataset model exp_reparam num_components"

python -m scripts.results --tboard-hparams hparams-artificial-discrete-categoricals.csv results-artificial-discrete-categoricals.csv \
    --average-reps --group-keys "dataset model exp_reparam num_components"

python -m scripts.results --tboard-hparams hparams-artificial-discrete-binomials.csv results-artificial-discrete-binomials.csv \
    --average-reps --group-keys "dataset model exp_reparam num_components"
"""


def build_command(r: pd.Series) -> str:
    hps = [
        'exp-alias',
        'dataset',
        'model',
        'num-components',
        'num-replicas',
        'depth',
        'optimizer',
        'batch-size',
        'learning-rate',
        'splines',
        'spline-order',
        'spline-knots',
        'init-method',
        'init-scale',
        'standardize',
        'compute-layer',
        'exp-reparam'
    ]
    int_hps = {
        'num-components',
        'num-replicas',
        'depth',
        'batch-size',
        'spline-order',
        'spline-knots'
    }
    bool_hps = {
        'splines': False,
        'exp-reparam': False
    }
    cmd = "python -m scripts.experiment"
    for hp in hps:
        if hp in r:
            hp_val = r[hp]
        else:
            hp_name = hp.replace('-', '_')
            hp_val = r[hp_name]
        if hp in int_hps:
            hp_val = int(hp_val)
        if hp in bool_hps:
            if bool(hp_val):
                cmd = f"{cmd} --{hp}"
                bool_hps[hp] = True
            continue
        dep_bool = True
        for bhp in bool_hps:
            if bhp in hp or bhp[:-1] in hp:
                dep_bool = bool_hps[bhp]
        if dep_bool:
            cmd = f"{cmd} --{hp} {hp_val}"
    return cmd


if __name__ == '__main__':
    args = parser.parse_args()
    groupby_keys = args.group_keys.split()
    metric = 'Best/Valid/' + args.metric

    assert (args.tboard_hparams and not args.tboard_path) or (not args.tboard_hparams and args.tboard_path), \
        "Exactly one between --tboard-hparams and --tboard-path must be specified"
    if args.tboard_path:
        df = retrieve_tboard_runs(args.tboard_path, metric)
    else:
        df = pd.read_csv(args.tboard_hparams)

    if args.average_reps:
        group_df = df.drop(
            ['region_graph', 'compute_layer', 'optimizer', 'init_method', 'exp_alias', 'git_rev_hash'],
            axis=1
        ).groupby(groupby_keys)
        agg_df = group_df.agg(['mean', 'std']).reset_index()
        group_df = df.groupby(by='dataset')
        num_groups = len(group_df)
        pairwise_pvalues = dict()
        max_num_inner_groups = -np.inf
        for i, data_df in group_df:
            inner_group_df = data_df.groupby(by=groupby_keys)
            num_inner_groups = len(inner_group_df)
            pairwise_vars = list(itertools.product(range(num_inner_groups), range(num_inner_groups)))
            inner_pairwise_pvalues = np.zeros(shape=(num_inner_groups, num_inner_groups), dtype=np.float64)
            ss = [r[1]['Best/Test/' + args.metric].tolist() for r in inner_group_df]
            for j, k in pairwise_vars:
                _, p = stats.mannwhitneyu(ss[j], ss[k], alternative='less' if args.best_lower else 'greater')
                inner_pairwise_pvalues[j, k] = p
            pairwise_pvalues[i] = inner_pairwise_pvalues
            if num_inner_groups > max_num_inner_groups:
                max_num_inner_groups = num_inner_groups
        df = agg_df
        rel = '<' if args.best_lower else '>'
        for j in range(max_num_inner_groups):
            pvalues = [
                np.concatenate([
                    pairwise_pvalues[i],
                    np.full(shape=(pairwise_pvalues[i].shape[0], max_num_inner_groups - pairwise_pvalues[i].shape[1]),
                            fill_value=np.nan)], axis=1)[:, j]
                for i in pairwise_pvalues.keys()
            ]
            pvalues = np.concatenate(pvalues, axis=0)
            df[f'Pvalue_{rel}{j}'] = pvalues.tolist()
        remove_columns = list(filter(lambda c: (c[0][0] != c[0][0].upper()) and c[0] not in groupby_keys, df.columns))
        df = df.drop(remove_columns, axis=1)
    else:
        group_df = df.groupby(groupby_keys)[metric]
        best_idx = group_df.idxmin() if args.best_lower else group_df.idxmax()
        df = df.loc[best_idx]

    # Round float columns
    float_cols = [
        c for (c, t) in df.dtypes.to_dict().items()
        if t in [float, np.dtype('float32'), np.dtype('float64')]
    ]
    df = df.round(dict((k, args.digits) for k in float_cols))

    # Save the best results
    df.to_csv(args.best_filepath, index=False)
    if not args.cmd_filepath:
        quit()

    # Save the list of commands of the best experimental outcomes
    with open(args.cmd_filepath, 'w') as f:
        for _, r in df.iterrows():
            cmd = build_command(r)
            f.write(f"{cmd}\n")
