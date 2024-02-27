import argparse
import gc
import os
from collections import defaultdict
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import rcParams
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

from graphics.utils import setup_tueplots
from pcs.models import PC
from scripts.utils import set_global_seed, setup_data_loaders, setup_model
from pcs.utils import num_parameters

parser = argparse.ArgumentParser(
    description="Benchmark for squared circuits"
)
parser.add_argument(
    '--datasets', type=str, required=True, help="The evaluation datasets, separated by space"
)
parser.add_argument(
    '--num-iterations', type=int, default=1, help="The number of iterations per benchmark"
)
parser.add_argument(
    '--burnin-iterations', type=int, default=1, help="Burnin iterations (additional to --num-iterations)"
)
parser.add_argument(
    '--device', type=str, default='cpu', help="The device id"
)
parser.add_argument(
    '--batch-size', type=int, default=512, help="The batch size to use"
)
parser.add_argument(
    '--num-components', type=int, default=512, help="The layer dimensionality"
)
parser.add_argument(
    '--min-bubble-radius', type=float, default=40.0, help="Bubble sizes minimum"
)
parser.add_argument(
    '--scale-bubble-radius', type=float, default=1.0, help="Bubble sizes scaler"
)
parser.add_argument(
    '--exp-bubble-radius', type=float, default=1.75, help="The exponent for computing the bubble sizes"
)
parser.add_argument(
    '--specific-hparams', type=str, default="",
    help="Specific hyperparameters (separated by space) per model (separated by dash) per dataset (separated by semicolon)"
)
#
# e.g., --specific-hparams "num_components=1024 batch_size=2048-num_components=1024 batch_size=512;num_components=128 batch_size=512-num_components=256 batch_size=512;num_components=32 batch_size=512-num_components=32 batch_size=512;num_components=512 batch_size=512-num_components=128 batch_size=512"
#
parser.add_argument(
    '--eval-backprop', action='store_true', default=False, help="Whether to benchmark also backpropagation"
)
parser.add_argument(
    '--seed', type=int, default=42, help="The seed for reproducibility"
)


def from_bytes_to_gib(bytes: int) -> float:
    return bytes / (1024.0 * 1024.0 * 1024.0)


def bubble_size(s: float, a: float = 0.0, m: float = 1.0, p: float = 2.0, inverse: bool = False) -> float:
    if inverse:
        return ((s - a) ** (1.0 / p)) / m
    return a + ((m * s) ** p)


def format_model_name(m: str, exp_reparam: bool = False) -> str:
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


def benchmark_model(
        model: PC,
        dataset: np.ndarray,
        batch_size: int,
        num_iterations: int,
        burnin_iterations: int,
        eval_pf: bool = False
):
    # Setup the data loader
    total_num_iterations = burnin_iterations + num_iterations
    while batch_size * total_num_iterations >= len(dataset):
        dataset = np.concatenate([dataset, dataset], axis=0)
    # assert batch_size * total_num_iterations < len(dataset), "Number of iterations is too large for this dataset and batch size"
    ordering = np.random.permutation(len(dataset))
    dataset = dataset[ordering]
    dataset = TensorDataset(torch.from_numpy(dataset[:batch_size * total_num_iterations]))
    data_loader = DataLoader(dataset, batch_size, drop_last=True)
    try:
        mu_time, mu_memory = run_benchmark(data_loader, model, burnin_iterations=burnin_iterations, eval_pf=eval_pf)
    except torch.cuda.OutOfMemoryError:
        mu_time, mu_memory = np.nan, np.nan
    return mu_time, mu_memory


def run_benchmark(data_loader: DataLoader, model: PC, burnin_iterations: int = 1, eval_pf: bool = False) -> Tuple[
    float, float]:
    if args.eval_backprop:
        # Setup losses and a dummy optimizer (only used to free gradient tensors)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    elapsed_times = list()
    gpu_memory_peaks = list()
    for batch_idx, batch in enumerate(data_loader):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        # Run GC manually and then disable it
        gc.collect()
        gc.disable()
        # Reset peak memory usage statistics
        torch.cuda.reset_peak_memory_stats(device)
        # torch.cuda.synchronize(device)  # Synchronize CUDA operations
        batch = batch.to(device)
        # torch.cuda.synchronize(device)  # Make sure the batch is already loaded (do not take into account this!)
        # start_time = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if eval_pf:
            lls = model.log_pf(return_input=False)
        else:
            lls = model.log_score(batch)
        if args.eval_backprop:
            loss = -lls.mean()
            loss.backward(retain_graph=False)  # Free the autodiff graph
        end.record()
        torch.cuda.synchronize(device)  # Synchronize CUDA Kernels before measuring time
        # end_time = time.perf_counter()
        gpu_memory_peaks.append(from_bytes_to_gib(torch.cuda.max_memory_allocated(device)))
        if args.eval_backprop:
            optimizer.zero_grad()  # Free gradients tensors
        gc.enable()  # Enable GC again
        gc.collect()  # Manual GC
        # elapsed_times.append(end_time - start_time)
        elapsed_times.append(start.elapsed_time(end) * 1e-3)

    # Discard burnin iterations and compute averages
    elapsed_times = elapsed_times[burnin_iterations:]
    gpu_memory_peaks = gpu_memory_peaks[burnin_iterations:]
    mu_time = np.mean(elapsed_times).item()
    print(f"Mean time: {mu_time} -- Std. time: {np.std(elapsed_times)}")
    mu_memory = np.mean(gpu_memory_peaks).item()
    return mu_time, mu_memory


def entry_uniform_hparams_configuration() -> Tuple[dict, list]:
    bench_results = defaultdict(dict)
    datasets = args.datasets.split()
    num_variables = list()
    for dataset in datasets:
        # Setup the data set
        metadata, (train_dataloader, valid_dataloader, test_dataloader) = setup_data_loaders(
            dataset, 'datasets', batch_size=1
        )
        x_data = train_dataloader.dataset.tensors[0].numpy()
        num_variables.append(x_data.shape[1])

        for idx, m in enumerate(models):
            print(f"Benchmarking {m} ...")
            model = setup_model(
                m, dataset_metadata=metadata, rg_type='random',
                rg_replicas=8, rg_depth=-1, num_components=num_components,
                compute_layer='cp', init_method='uniform', init_scale=1.0,
                seed=args.seed
            )

            num_params = num_parameters(model)
            print(f"Model architecture:\n{model}")
            print(f"Number of parameters: {num_params}")
            model.to(device)

            if 'c' not in bench_results[m]:
                bench_results[m]['c'] = dict()
                bench_results[m]['Z'] = dict()

            mu_time, mu_memory = benchmark_model(
                model, x_data, batch_size=batch_size,
                num_iterations=args.num_iterations, burnin_iterations=args.burnin_iterations,
                eval_pf=False
            )
            bench_results[m]['c'][dataset] = (mu_time, mu_memory)

            mu_time, mu_memory = benchmark_model(
                model, x_data, batch_size=batch_size,
                num_iterations=args.num_iterations, burnin_iterations=args.burnin_iterations,
                eval_pf=True
            )
            bench_results[m]['Z'][dataset] = (mu_time, mu_memory)

            del model
    return bench_results, num_variables


def entry_specific_hparams_configuration(hparams_conf: dict) -> Tuple[dict, list]:
    log_likelihoods = {
        'gas': {'MonotonicPC': 5.56, 'BornPC': 10.98},
        'hepmass': {'MonotonicPC': -22.45, 'BornPC': -20.41},
        'miniboone': {'MonotonicPC': -32.11, 'BornPC': -26.92},
        'bsds300': {'MonotonicPC': 123.30, 'BornPC': 128.38},
    }
    bench_results = defaultdict(dict)
    datasets = args.datasets.split()
    num_variables = list()
    for dataset, hps_conf in zip(datasets, hparams_conf):
        # Setup the data set
        metadata, (train_dataloader, valid_dataloader, test_dataloader) = setup_data_loaders(
            dataset, 'datasets', batch_size=1
        )
        x_data = train_dataloader.dataset.tensors[0].numpy()
        num_variables.append(x_data.shape[1])

        for idx, (m, hps) in enumerate(zip(models, hps_conf)):
            print(f"Benchmarking {m} ...")
            other_hps = dict()
            if dataset == 'bsds300':
                other_hps['splines'] = True
                other_hps['spline_knots'] = 512

            model = setup_model(
                m, dataset_metadata=metadata, rg_type='random',
                rg_replicas=8, rg_depth=-1, num_components=int(hps['num_components']),
                compute_layer='cp', init_method='uniform', init_scale=1.0,
                seed=args.seed, **other_hps
            )

            num_params = num_parameters(model)
            print(f"Model architecture:\n{model}")
            print(f"Number of parameters: {num_params}")
            model.to(device)

            if 'c' not in bench_results[m]:
                bench_results[m]['c'] = dict()
                bench_results[m]['Z'] = dict()
                bench_results[m]['ll'] = dict()
            bench_results[m]['ll'][dataset] = log_likelihoods[dataset][m]

            mu_time, mu_memory = benchmark_model(
                model, x_data, batch_size=int(hps['batch_size']),
                num_iterations=args.num_iterations, burnin_iterations=args.burnin_iterations,
                eval_pf=False
            )
            bench_results[m]['c'][dataset] = (mu_time, mu_memory)

            mu_time, mu_memory = benchmark_model(
                model, x_data, batch_size=int(hps['batch_size']),
                num_iterations=args.num_iterations, burnin_iterations=args.burnin_iterations,
                eval_pf=True
            )
            bench_results[m]['Z'][dataset] = (mu_time, mu_memory)

            del model
    return bench_results, num_variables


if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size
    num_components = args.num_components
    models = ['MonotonicPC', 'BornPC']

    # Set device and the seed
    device = torch.device(args.device)
    set_global_seed(args.seed)
    if not args.eval_backprop:
        torch.set_grad_enabled(False)


    def _bubble_size(s, inverse=False):
        return bubble_size(
            s, a=args.min_bubble_radius,
            m=args.scale_bubble_radius, p=args.exp_bubble_radius,
            inverse=inverse
        )


    nrows, ncols = 1, 2
    setup_tueplots(nrows, ncols, hw_ratio=0.5)
    fig, ax = plt.subplots(nrows, ncols, sharey=True)

    if args.specific_hparams:
        hps_per_dataset = args.specific_hparams.split(';')
        # I have no idea what I am doing, it's functional magic
        hps_per_dataset_model = list(map(lambda hps: hps.split('-'), hps_per_dataset))
        hparams_conf = list(
            map(lambda hps: list(map(lambda h: dict(map(lambda x: tuple(x.split('=')), h.split(' '))), hps)),
                hps_per_dataset_model))
        print(hparams_conf)
        bench_results, num_variables = entry_specific_hparams_configuration(hparams_conf)
    else:
        bench_results, num_variables = entry_uniform_hparams_configuration()

    print(bench_results)
    print(f"Plotting results")

    scatter_plots = dict()
    for idx, (m, br) in enumerate(bench_results.items()):
        eval_c_results = list(map(lambda d: br['c'][d], br['c'].keys()))
        eval_Z_results = list(map(lambda d: br['Z'][d], br['Z'].keys()))
        print(eval_c_results)
        print(eval_Z_results)

        if args.specific_hparams:
            eval_ll_results = list(map(lambda d: br['ll'][d], br['ll'].keys()))
            print(eval_ll_results)

        eval_c_results_time = list(map(lambda t: t[0], eval_c_results))
        eval_Z_results_time = list(map(lambda t: t[0], eval_Z_results))

        if m == 'MonotonicPC':
            desc_c = r'$c(\mathbf{x})$'
            desc_Z = r'$Z = \int c(\mathbf{x}) \mathrm{d}\mathbf{x}$'
        elif m == 'BornPC':
            desc_c = r'$c^2(\mathbf{x})$'
            desc_Z = r'$Z = \int c^2(\mathbf{x}) \mathrm{d}\mathbf{x}$'
        else:
            assert False

        sc_c = ax[0].scatter(
            num_variables, eval_c_results_time,
            color=f'C{idx}', alpha=.5, s=list(map(lambda t: _bubble_size(t[1]), eval_c_results)),
            marker='o', label=desc_c
        )
        ax[0].scatter(
            num_variables, eval_c_results_time,
            color='k', alpha=.6, s=1, marker='o'
        )
        if args.specific_hparams:
            for xi, yi, ll in zip(num_variables, eval_c_results_time, eval_ll_results):
                xytext = (0, 1) if m == 'BornPC' else (0, -6)
                ax[0].annotate(f"{ll}", xy=(xi, yi), fontsize=6, xytext=xytext, textcoords='offset points')

        sc_Z = ax[1].scatter(
            num_variables, eval_Z_results_time,
            color=f'C{idx}', alpha=.5, s=list(map(lambda t: _bubble_size(t[1]), eval_Z_results)),
            marker='o', label=desc_Z
        )
        ax[1].scatter(
            num_variables, eval_Z_results_time,
            color='k', alpha=.6, s=1, marker='o'
        )
        if args.specific_hparams:
            for xi, yi, ll in zip(num_variables, eval_Z_results_time, eval_ll_results):
                xytext = (0, 1) if m == 'BornPC' else (0, -6)
                ax[1].annotate(f"{ll}", xy=(xi, yi), fontsize=6, xytext=xytext, textcoords='offset points')

        scatter_plots[m] = {'c': sc_c, 'Z': sc_Z}

    ax[0].set_ylabel('Time per batch ($s$)')
    ax[0].annotate(r'$|\mathbf{X}|$', xy=(1, 0), xytext=(1, -1 * rcParams['xtick.major.pad']), ha='right', va='top',
                   xycoords='axes fraction', textcoords='offset points')
    ax[1].annotate(r'$|\mathbf{X}|$', xy=(1, 0), xytext=(1, -1 * rcParams['xtick.major.pad']), ha='right', va='top',
                   xycoords='axes fraction', textcoords='offset points')
    ax[0].set_axisbelow(True)
    ax[1].set_axisbelow(True)
    # ax[0].set_xscale('log')
    # ax[1].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    # ax[1].set_xticks([2 * (10 ** 5), 10 ** 6])
    ax[0].margins(x=0.2, y=0.275)
    ax[1].margins(x=0.2, y=0.275)
    #ax[0].set_ylim(bottom=-0.05)
    #ax[1].set_ylim(bottom=-0.05)
    ax[0].grid(linestyle='--', alpha=0.3, linewidth=.5)
    ax[1].grid(linestyle='--', alpha=0.3, linewidth=.5)
    c_legend_loc = 'upper center' if args.specific_hparams else 'upper left'
    c_legend = ax[0].legend(
        loc=c_legend_loc,
        # bbox_to_anchor=(1.0, 1.0),
        labelspacing=0.4,
        framealpha=0.4
    )

    for i in range(len(c_legend.legend_handles)):
        c_legend.legend_handles[i].set_sizes([20])
    ax[0].add_artist(c_legend)
    z_legend_loc = 'upper center' if args.specific_hparams else 'upper left'
    z_legend = ax[1].legend(loc=z_legend_loc, framealpha=0.4)
    ax[0].text(-0.62, 0.45, "GPU Memory (GiB)", rotation=90, va='center', transform=ax[0].transAxes)
    ax[0].legend(
        loc='upper right', bbox_to_anchor=(-0.24, 1.05),
        labelspacing=1.4 if args.specific_hparams else 2.2, frameon=False,
        *scatter_plots[models[-1]]['Z'].legend_elements(
            prop='sizes', func=lambda s: _bubble_size(s, inverse=True),
            alpha=0.6, fmt="{x:.0f}", num=4
        )
    )
    os.makedirs(os.path.join('figures', 'benchmarks'), exist_ok=True)
    filename = 'benchmark-vars-hparams' if args.specific_hparams else 'benchmark-vars'
    if args.eval_backprop:
        filename = f'{filename}-backprop'
    plt.savefig(os.path.join('figures', 'benchmarks', f'{filename}.pdf'))
