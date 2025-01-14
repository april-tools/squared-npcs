import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from graphics.utils import setup_tueplots
from pcs.initializers import INIT_METHODS
from pcs.layers import COMPUTE_LAYERS
from pcs.optimizers import OPTIMIZERS_NAMES
from region_graph import REGION_GRAPHS
from scripts.utils import setup_experiment_path, build_run_id, format_model

parser = argparse.ArgumentParser(
    description="Plot the distribution of the weights",
)
parser.add_argument('path', default='checkpoints', type=str, help="The checkpoints root path")
parser.add_argument('--dataset', type=str, default="", help="Dataset name")
parser.add_argument('--model', default="", help="The model name")
parser.add_argument('--region-graph', type=str, choices=REGION_GRAPHS, default=REGION_GRAPHS[0],
                    help="The region graph to use")
parser.add_argument('--num-replicas', default=1, type=int, help="Number of replicas")
parser.add_argument('--num-components', default=2, type=int, help="Number of components")
parser.add_argument('--depth', default=1, type=int,
                    help="The detph of the region graph. If negative, the it is the maximum depth allowed")
parser.add_argument('--exp-alias', default="", help="The experiment alias, if any")
parser.add_argument('--optimizer', choices=OPTIMIZERS_NAMES, default=OPTIMIZERS_NAMES[0], help="Optimiser to use")
parser.add_argument('--compute-layer', choices=COMPUTE_LAYERS, default=COMPUTE_LAYERS[0], help="The compute layer")
parser.add_argument('--learning-rate', default=0.01, type=float, help="Learning rate")
parser.add_argument('--batch-size', default=256, type=int, help="Batch size")
parser.add_argument('--splines', action='store_true', default=False, help="Whether to enable splines")
parser.add_argument('--spline-order', type=int, default=2, help="The B-spline order")
parser.add_argument('--spline-knots', type=int, default=8, help="The number of uniformly-chosen knots within the data")
parser.add_argument('--exp-reparam', action='store_true', default=False,
                    help="Whether to reparameterize the parameters of BornPCs via exponentiation")
parser.add_argument('--weight-decay', default=0.0, type=float, help="The L2 factor or weight decay")
parser.add_argument('--init-method', choices=INIT_METHODS, default=INIT_METHODS[0], help="Parameters initialisers")
parser.add_argument('--filename', type=str, required=True,
                    help="The name of the file where to save the plot")
parser.add_argument('--legend', action='store_true', default=False)


"""
python -m scripts.plots.wdist checkpoints/uci-data-weights --dataset power --model MonotonicPC \
    --region-graph random --num-replicas 8 --depth -1 --num-components 512 --splines --spline-order 2 \
    --spline-knots 512 --learning-rate 0.01 --batch-size 512 --init-method uniform --filename mono-power.pdf --legend
python -m scripts.plots.wdist checkpoints/uci-data-weights --dataset power --model BornPC --region-graph random \
    --num-replicas 8 --depth -1 --num-components 512 --splines --spline-order 2 --spline-knots 512 \
    --learning-rate 0.01 --batch-size 512 --init-method uniform --exp-alias non-monotonic \
    --filename squared-power.pdf --legend
python -m scripts.plots.wdist checkpoints/uci-data-weights --dataset gas --model MonotonicPC \
    --region-graph random --num-replicas 8 --depth -1 --num-components 512 --splines --spline-order 2 \
    --spline-knots 512 --learning-rate 0.01 --batch-size 512 --init-method uniform --filename mono-gas.pdf --legend
python -m scripts.plots.wdist checkpoints/uci-data-weights --dataset gas --model BornPC --region-graph random \
    --num-replicas 8 --depth -1 --num-components 512 --splines --spline-order 2 --spline-knots 512 \
    --learning-rate 0.01 --batch-size 512 --init-method uniform --exp-alias non-monotonic \
    --filename squared-gas.pdf --legend
python -m scripts.plots.wdist checkpoints/uci-data-weights --dataset hepmass --model MonotonicPC \
    --region-graph random --num-replicas 8 --depth -1 --num-components 128 --splines --spline-order 2 \
    --spline-knots 512 --learning-rate 0.01 --batch-size 512 --init-method uniform --filename mono-hepmass.pdf --legend
python -m scripts.plots.wdist checkpoints/uci-data-weights --dataset hepmass --model BornPC --region-graph random \
    --num-replicas 8 --depth -1 --num-components 128 --splines --spline-order 2 --spline-knots 512 \
    --learning-rate 0.01 --batch-size 512 --init-method uniform --exp-alias non-monotonic \
    --filename squared-hepmass.pdf --legend
python -m scripts.plots.wdist checkpoints/uci-data-weights --dataset miniboone --model MonotonicPC --region-graph random \
    --num-replicas 8 --depth -1 --num-components 128 --splines --spline-order 2 --spline-knots 512 \
    --learning-rate 0.01 --batch-size 512 --init-method uniform --filename mono-miniboone.pdf --legend
python -m scripts.plots.wdist checkpoints/uci-data-weights --dataset miniboone --model BornPC --region-graph random \
    --num-replicas 8 --depth -1 --num-components 128 --splines --spline-order 2 --spline-knots 512 \
    --learning-rate 0.01 --batch-size 512 --init-method uniform --exp-alias non-monotonic \
    --filename squared-miniboone.pdf --legend
python -m scripts.plots.wdist checkpoints/uci-data-weights --dataset bsds300 --model MonotonicPC --region-graph random \
    --num-replicas 8 --depth -1 --num-components 64 --splines --spline-order 2 --spline-knots 512 \
    --learning-rate 0.01 --batch-size 512 --init-method uniform --filename mono-bsds300.pdf --legend
python -m scripts.plots.wdist checkpoints/uci-data-weights --dataset bsds300 --model BornPC --region-graph random \
    --num-replicas 8 --depth -1 --num-components 64 --splines --spline-order 2 --spline-knots 512 \
    --learning-rate 0.01 --batch-size 512 --init-method uniform --exp-alias non-monotonic \
    --filename squared-bsds300.pdf --legend
python -m scripts.plots.wdist checkpoints/gpt2-commongen-ihmm-weights --dataset gpt2_commongen --model MonotonicPC \
    --region-graph linear-vtree --num-components 128 --learning-rate 0.01 --batch-size 4096 --init-method uniform \
    --filename mono-gpt2.pdf --legend
python -m scripts.plots.wdist checkpoints/gpt2-commongen-ihmm-weights --dataset gpt2_commongen --model BornPC \
    --region-graph linear-vtree --num-components 128 --learning-rate 0.01 --batch-size 4096 --init-method uniform \
    --filename squared-gpt2.pdf --legend
"""


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    # Load state dictionary
    args = parser.parse_args()
    path = setup_experiment_path(
        args.path, args.dataset, args.model, args.exp_alias, trial_id=build_run_id(args))
    sd = torch.load(os.path.join(path, 'model.pt'), map_location='cpu')['weights']
    print(sd.keys())

    # Concatenate weights in a large vector
    ws = list()
    for k in sd.keys():
        # Select the parameters of CP layers only
        if 'layer' in k and 'weight' in k and 'input' not in k and 'mixture' not in k:
            w = sd[k]
            if 'Born' in args.model:  # Perform squaring
                if len(w.shape) == 3:  # CP layer
                    w = torch.einsum('fki,fkj->fkij', w, w)
                else:
                    assert False, "This should not happen :("
            ws.append(w.flatten().numpy())
    ws = np.concatenate(ws, axis=0)

    # Preprocess the weights, and set some flags
    if 'Mono' in args.model:
        mb = np.quantile(ws, q=[0.99], method='lower')
        ws = ws[ws <= mb]
        ws = np.exp(ws)
        hcol = 'C0'
    elif 'Born' in args.model:
        ma, mb = np.quantile(ws, q=[0.005, 0.995], method='lower')
        ws = ws[(ws >= ma) & (ws <= mb)]
        hcol = 'C1'
    print(ws.shape)

    # Compute and plot the instogram
    setup_tueplots(1, 1, rel_width=0.25, hw_ratio=1.0)
    hlabel = f'{format_model(args.model)}'
    plt.hist(ws, bins=64, color=hcol, label=hlabel)
    plt.yscale('log')
    if args.legend:
        plt.legend()

    os.makedirs(os.path.join('figures', 'model-weights'), exist_ok=True)
    plt.savefig(os.path.join('figures', 'model-weights', args.filename), dpi=1200)
