import argparse
import gc
import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import rcParams
from torch.utils.data import DataLoader, TensorDataset

from graphics.utils import setup_tueplots
from pcs.models import PC
from scripts.utils import set_global_seed, setup_data_loaders, setup_model
from pcs.utils import num_parameters

parser = argparse.ArgumentParser(
    description="Benchmark for squared circuits"
)
parser.add_argument(
    'dataset', type=str, help="The evaluation dataset"
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
    '--batch-sizes', type=str, default="128 256 512 1024", help="A list of batch sizes separated by space"
)
parser.add_argument(
    '--base-batch-size', type=int, default=512, help="The default batch size"
)
parser.add_argument(
    '--num-components', type=str, default="64 128 256 512", help="A list of layer dimensionality separated by space"
)
parser.add_argument(
    '--base-num-components', type=int, default=256, help="The default number of components"
)
parser.add_argument(
    '--min-bubble-radius', type=float, default=20.0, help="Bubble sizes minimum"
)
parser.add_argument(
    '--scale-bubble-radius', type=float, default=1.0, help="Bubble sizes scaler"
)
parser.add_argument(
    '--exp-bubble-radius', type=float, default=1.75, help="The exponent for computing the bubble sizes"
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
    #assert batch_size * total_num_iterations < len(dataset), "Number of iterations is too large for this dataset and batch size"
    ordering = np.random.permutation(len(dataset))
    dataset = dataset[ordering]
    dataset = TensorDataset(torch.from_numpy(dataset[:batch_size * total_num_iterations]))
    data_loader = DataLoader(dataset, batch_size, drop_last=True)
    try:
        mu_time, mu_memory = run_benchmark(data_loader, model, burnin_iterations=burnin_iterations, eval_pf=eval_pf)
    except torch.cuda.OutOfMemoryError:
        mu_time, mu_memory = np.nan, np.nan
    return mu_time, mu_memory


def run_benchmark(data_loader: DataLoader, model: PC, burnin_iterations: int = 1, eval_pf: bool = False) -> Tuple[float, float]:
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
        #torch.cuda.synchronize(device)  # Synchronize CUDA operations
        batch = batch.to(device)
        #torch.cuda.synchronize(device)  # Make sure the batch is already loaded (do not take into account this!)
        #start_time = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if eval_pf:
            lls = model.log_pf(return_input=False)
        else:
            lls = model.log_score(batch)
        end.record()
        torch.cuda.synchronize(device)     # Synchronize CUDA Kernels before measuring time
        #end_time = time.perf_counter()
        gpu_memory_peaks.append(from_bytes_to_gib(torch.cuda.max_memory_allocated(device)))
        gc.enable()            # Enable GC again
        gc.collect()           # Manual GC
        #elapsed_times.append(end_time - start_time)
        elapsed_times.append(start.elapsed_time(end) * 1e-3)

    # Discard burnin iterations and compute averages
    elapsed_times = elapsed_times[burnin_iterations:]
    gpu_memory_peaks = gpu_memory_peaks[burnin_iterations:]
    mu_time = np.mean(elapsed_times).item()
    print(f"Mean time: {mu_time} -- Std. time: {np.std(elapsed_times)}")
    mu_memory = np.mean(gpu_memory_peaks).item()
    return mu_time, mu_memory


if __name__ == '__main__':
    args = parser.parse_args()
    batch_sizes = sorted(map(int, args.batch_sizes.split()))
    num_components = sorted(map(int, args.num_components.split()))
    models = ['BornPC', 'BornPC']
    settings = [{'eval_pf': False}, {'eval_pf': True}]

    # Set device and the seed
    device = torch.device(args.device)
    set_global_seed(args.seed)
    torch.set_grad_enabled(False)

    # Setup the data set
    metadata, (train_dataloader, valid_dataloader, test_dataloader) = setup_data_loaders(
        args.dataset, 'datasets', 100
    )
    dataset = train_dataloader.dataset.tensors[0].numpy()

    nrows, ncols = 1, 2
    setup_tueplots(nrows, ncols)
    fig, ax = plt.subplots(nrows, ncols, sharey=True)
    def _bubble_size(s, inverse=False):
        return bubble_size(
            s, a=args.min_bubble_radius,
            m=args.scale_bubble_radius, p=args.exp_bubble_radius,
            inverse=inverse
        )

    scatter_plots = dict()
    markers = ['o', 'o', 'o', 's', '^', 'D', 'h']
    for idx, (m, ss) in enumerate(zip(models, settings)):
        conf = dict()
        conf['model_name'] = m

        print(f"Benchmarking {m} by varying the batch size ...")
        model = setup_model(
            dataset_metadata=metadata, rg_type='random',
            rg_replicas=32, rg_depth=-1, num_components=args.base_num_components,
            compute_layer='cp', init_method='uniform', init_scale=1.0,
            seed=args.seed, **conf
        )
        num_params = num_parameters(model)
        print(f"Model architecture:\n{model}")
        print(f"Number of parameters: {num_params}")
        model.to(device)
        bench_bs_results = list()
        for bs in batch_sizes:
            mu_time, mu_memory = benchmark_model(
                model, dataset, batch_size=bs,
                num_iterations=args.num_iterations, burnin_iterations=args.burnin_iterations,
                **ss
            )
            bench_bs_results.append((mu_time, mu_memory))

        print(f"Benchmarking {m} by varying the number of components ...")
        bench_nc_results = list()
        for nc in num_components:
            del model
            model = setup_model(
                dataset_metadata=metadata, rg_type='random',
                rg_replicas=32, rg_depth=-1, num_components=nc,
                compute_layer='cp', init_method='uniform', init_scale=1.0,
                seed=args.seed, **conf
            )
            num_params = num_parameters(model)
            print(f"Model architecture:\n{model}")
            print(f"Number of parameters: {num_params}")
            model.to(device)
            mu_time, mu_memory = benchmark_model(
                model, dataset, batch_size=args.base_batch_size,
                num_iterations=args.num_iterations, burnin_iterations=args.burnin_iterations,
                **ss
            )
            bench_nc_results.append((mu_time, mu_memory))
        del model

        print(f"Plotting results for {m}")
        #desc = format_model_name(m)
        desc = r"$Z \ (\pm^2)$" if ss['eval_pf'] else r"$c(\mathbf{X}) \ (\pm)$"
        bench_bs_results = list(filter(lambda t: np.isfinite(t[0]), bench_bs_results))
        bench_nc_results = list(filter(lambda t: np.isfinite(t[0]), bench_nc_results))
        print(bench_bs_results)
        print(bench_nc_results)
        sc_bs = ax[0].scatter(
            batch_sizes[:len(bench_bs_results)], list(map(lambda t: t[0], bench_bs_results)),
            color=f'C{idx}', alpha=.5, s=list(map(lambda t: _bubble_size(t[1]), bench_bs_results)),
            marker='o', label=desc
        )
        ax[0].scatter(
            batch_sizes[:len(bench_bs_results)], list(map(lambda t: t[0], bench_bs_results)),
            color='k', alpha=.6, s=1, marker=markers[idx]
        )
        sc_nc = ax[1].scatter(
            num_components, list(map(lambda t: t[0], bench_nc_results)),
            color=f'C{idx}', alpha=.5, s=list(map(lambda t: _bubble_size(t[1]), bench_nc_results)),
            marker='o', label=desc
        )
        ax[1].scatter(
            num_components, list(map(lambda t: t[0], bench_nc_results)),
            color='k', alpha=.6, s=1, marker=markers[idx]
        )
        scatter_plots[f"{m}-{ss['eval_pf']}"] = {'bs': sc_bs, 'nc': sc_nc}

    print(scatter_plots)

    ax[0].set_ylabel('Time per batch ($s$)')
    ax[0].annotate('BS', xy=(1, 0), xytext=(1, -1.5 * rcParams['xtick.major.pad']), ha='right', va='top',
                   xycoords='axes fraction', textcoords='offset points')
    ax[1].annotate(r'$K$', xy=(1, 0), xytext=(1, -1.5 * rcParams['xtick.major.pad']), ha='right', va='top',
                   xycoords='axes fraction', textcoords='offset points')
    ax[0].set_axisbelow(True)
    ax[1].set_axisbelow(True)
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    #ax[1].set_xticks([2 * (10 ** 5), 10 ** 6])
    ax[0].margins(x=0.175, y=0.325)
    ax[1].margins(x=0.175, y=0.325)
    ax[0].set_ylim(bottom=-0.05)
    ax[1].set_ylim(bottom=-0.05)
    ax[0].grid(linestyle='--', alpha=0.3, linewidth=.5)
    ax[1].grid(linestyle='--', alpha=0.3, linewidth=.5)
    models_legend = ax[0].legend(
        loc='upper left',
        #bbox_to_anchor=(1.0, 1.0),
        labelspacing=0.4
    )
    for i in range(len(models_legend.legend_handles)):
        models_legend.legend_handles[i].set_sizes([20])
    ax[0].add_artist(models_legend)
    ax[0].text(-0.62, 0.45, "GPU Memory (GiB)", rotation=90, va='center', transform=ax[0].transAxes)
    ax[0].legend(
        loc='upper right', bbox_to_anchor=(-0.24, 1.05),
        labelspacing=2.2, frameon=False,
        *scatter_plots[f"{models[0]}-{False}"]['bs'].legend_elements(
            prop='sizes', func=lambda s: _bubble_size(s, inverse=True),
            alpha=0.6, fmt="{x:.0f}", num=4
        ),
        handletextpad=1.0
    )
    os.makedirs(os.path.join('figures', 'benchmarks'), exist_ok=True)
    plt.savefig(os.path.join('figures', 'benchmarks', f'benchmark-{args.dataset}.pdf'))

