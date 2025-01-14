import os
from copy import copy

import wandb
import argparse
from argparse import Namespace

from datasets.loaders import ALL_DATASETS
from pcs.hmm import HMM_MODELS
from pcs.initializers import INIT_METHODS
from pcs.layers import COMPUTE_LAYERS
from pcs.optimizers import OPTIMIZERS_NAMES
from pcs.models import PCS_MODELS
from region_graph import REGION_GRAPHS
from scripts.engine import Engine

parser = argparse.ArgumentParser(
    description="Experiment Launcher"
)
parser.add_argument('--seed', default=123, type=int, help="Seed user for random states")
parser.add_argument('--device', default='cpu', type=str, help="The Torch device to use")
parser.add_argument('--data-path', default='datasets', type=str, help="The data root path")
parser.add_argument('--tboard-path', default='', type=str, help="The Tensorboard path, empty to disable")
parser.add_argument('--log-distribution', action='store_true', default=False,
                    help="Whether to log the learned distribution")
parser.add_argument('--log-frequency', default=100, type=int,
                    help="The frequency for logging distributions")
parser.add_argument('--wandb-path', default='', type=str, help="The W&B path, empty to disable")
parser.add_argument('--wandb-project', default='born-pcs', type=str, help="The W&B project")
parser.add_argument('--wandb-sweeps', type=int, default=0, help="How many hyperparameters to sweep, 0 to disable")
parser.add_argument('--wandb-sweep-id', default='', type=str, help="The W&B sweep id")
parser.add_argument('--checkpoint-path', default='checkpoints', type=str, help="The checkpoints path")
parser.add_argument('--exp-alias', default='', type=str, help="Additional experiment alias, if any")
parser.add_argument('--save-checkpoint', action='store_true', default=False, help="Whether to save checkpoints")
parser.add_argument('--dataset', choices=ALL_DATASETS, required=True, help="Dataset name")
parser.add_argument('--num-samples', default=10000, type=int, help="Number of samples for artificial datasets")
parser.add_argument('--standardize', action='store_true', default=False, help="Whether to standardize the dataset")
parser.add_argument('--dequantize', action='store_true', default=False, help="Whether to dequantize images")
parser.add_argument('--discretize', action='store_true', default=False, help="Whether to discretize artificial data")
parser.add_argument('--discretize-unique', action='store_true', default=False,
                    help="Whether to ensure discretized artificial samples are unique")
parser.add_argument('--discretize-bins', default=32, type=int, help="Number of discretization bins")
parser.add_argument('--shuffle-bins', action='store_true', default=False,
                    help="Whether to shuffle bins, in case of using discretized artificial data")
parser.add_argument('--model', choices=PCS_MODELS + HMM_MODELS + ['NICE', 'MAF', 'NSF'],
                    required=True, help="The model name")
parser.add_argument('--num-workers', type=int, default=0, help="The number of data loader workers")
parser.add_argument('--binomials', action='store_true', default=False,
                    help="Whether to use binomial instead of categoricals, in case of discrete data")
parser.add_argument('--input-mixture', action='store_true', default=False,
                    help="Whether to add a mixture layers just after the input layer")
parser.add_argument('--num-components', default=2, type=int, help="Number of components")
parser.add_argument('--num-replicas', default=1, type=int, help="Number of replicas")
parser.add_argument('--depth', default=1, type=int,
                    help="The detph of the region graph. If negative, the it is the maximum depth allowed")
parser.add_argument('--compute-layer', choices=COMPUTE_LAYERS, default=COMPUTE_LAYERS[0], help="The compute layer")
parser.add_argument('--multivariate', action='store_true', default=False, help="Whether to allow multivariate dists")
parser.add_argument('--region-graph', type=str, choices=REGION_GRAPHS, default=REGION_GRAPHS[0],
                    help="The region graph to use")
parser.add_argument('--splines', action='store_true', default=False, help="Whether to enable splines")
parser.add_argument('--spline-order', type=int, default=2, help="The B-spline order")
parser.add_argument('--spline-knots', type=int, default=8, help="The number of uniformly-chosen knots within the data")
parser.add_argument('--spline-lsq', action='store_true', default=False,
                    help="Whether to initialize the splines via least squares")
parser.add_argument('--spline-lsq-noise', default=1e-1, type=float,
                    help="The amount of noise to apply relative to the least squares initialization method")
parser.add_argument('--exp-reparam', action='store_true', default=False,
                    help="Whether to reparameterize the parameters of BornPCs via exponentiation")
parser.add_argument('--l2norm', action='store_true', default=False,
                    help="Wether to apply L2 norm to the parameters (valid only for HMMs)")
parser.add_argument('--init-method', choices=INIT_METHODS, default=INIT_METHODS[0], help="Parameters initialisers")
parser.add_argument('--init-scale', type=float, default=1.0, help="The initialization scale for the layers")
parser.add_argument('--optimizer', choices=OPTIMIZERS_NAMES, default=OPTIMIZERS_NAMES[0], help="Optimiser to use")
parser.add_argument('--reduce-lr-plateau', action='store_true', default=False, help="Whether to reduce LR at plateau")
parser.add_argument('--reduce-lr-patience', default=10, type=int, help="The patience related to reducing the LR")
parser.add_argument('--patience-threshold', type=float, default=1e-3, help="The minimum relative improvent of patience")
parser.add_argument('--step-lr-decay', action='store_true', default=False, help="Whether to reduce LR at steps")
parser.add_argument('--step-size-lr-decay', type=int, help="The number of steps before decaying the LR")
parser.add_argument('--amount-lr-decay', type=float, default=0.5, help="The multiplicative LR decay")
parser.add_argument('--early-stop-loss', action='store_true', default=False,
                    help="Whether to early stop and save checkpoints based on training loss."
                         "If false then use validation metrics.")
parser.add_argument('--early-stop-patience', default=50, type=int, help="The patience epochs for early stopping")
parser.add_argument('--num-epochs', default=500, type=int, help="Number of epochs")
parser.add_argument('--batch-size', default=256, type=int, help="Batch size")
parser.add_argument('--learning-rate', default=0.01, type=float, help="Learning rate")
parser.add_argument('--decay1', default=0.9, type=float, help="Decay rate for the first moment estimate in Adam")
parser.add_argument('--decay2', default=0.999, type=float, help="Decay rate for second moment estimate in Adam")
parser.add_argument('--momentum', default=0.9, type=float, help="Momentum parameter in SGD")
parser.add_argument('--weight-decay', default=0.0, type=float, help="The L2 factor or weight decay")
parser.add_argument('--load-checkpoint', action='store_true', default=False, help="Whether to load a checkpoint")
parser.add_argument('--load-checkpoint-path', type=str, default='',
                    help="Alternative checkpoint path respect to --checkpoint-path")
parser.add_argument(
    '--checkpoint-hparams', default='', type=str,
    help="The optional hyperparameters of the checkpoint to load, e.g., 'model=MonotonicPC;init-method=normal'")
parser.add_argument('--verbose', action='store_true', default=False, help="Whether to plot stuff")
parser.add_argument('--dtype', default='float64', choices=['float32', 'float64'], help="The default Torch dtype to use")
parser.add_argument(
    '--show-bar', action='store_true', default=False,
    help="Whether to show the progress bar for each training epoch"
)


def run_engine(args: Namespace):
    engine = Engine(args)
    engine.run()
    engine.shutdown()


def run_engine_sweep():
    group = f'{args.dataset}-{args.model}'
    if args.exp_alias:
        group = f'{group}-{args.exp_alias}'
    os.makedirs(args.wandb_path, exist_ok=True)
    wandb.init(
        project=args.wandb_project,
        dir=args.wandb_path,
        group=group)
    wandb_config = wandb.config.items()
    next_args = copy(args)
    for hp_name, hp_value in wandb_config:
        if hp_name in args:
            next_args.__setattr__(hp_name, hp_value)
    run_engine(next_args)
    wandb.finish(quiet=True)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.wandb_path and args.wandb_sweeps > 0:
        wandb.agent(
            sweep_id=f"{args.wandb_project}/{args.wandb_sweep_id}",
            function=run_engine_sweep,
            count=args.wandb_sweeps)
    else:
        run_engine(args)
