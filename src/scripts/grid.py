from typing import Iterator, List, Optional, Tuple
from collections import defaultdict

import json
import itertools
import subprocess
import multiprocessing
import argparse

device_ids_cycle_g: Optional[Iterator[int]] = None


def expand_hparams_grid(hps_grid: dict, common_hps_grid: dict) -> List[dict]:
    grid = common_hps_grid.copy()
    grid.update(hps_grid)
    for k in grid.keys():
        v = grid[k]
        if not isinstance(v, list):
            grid[k] = [v]
    return [dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())]


def model_hparams(hps_grid: dict, common_hps_grid: dict) -> List[Tuple[Optional[str], dict]]:
    if not hps_grid:
        return [(None, common_hps_grid)]
    hps_grid_values = hps_grid.values()
    if all(isinstance(v, dict) for v in hps_grid_values):
        aliased_hps = []
        for ea, hps_specs in hps_grid.items():
            hps = expand_hparams_grid(hps_specs, common_hps_grid)
            aliased_hps.extend((ea, h) for h in hps)
        return aliased_hps
    assert not any(isinstance(v, dict) for v in hps_grid_values)
    hps = expand_hparams_grid(hps_grid, common_hps_grid)
    return [(None, h) for h in hps]


def build_command_string(dataset: str, model: str, hp: dict) -> str:
    c = 'python src/scripts/experiment.py'
    c = f'{c} --dataset {dataset} --model {model}'
    for k, v in hp.items():
        if isinstance(v, bool):
            if v:
                c = f'{c} --{k}'
        elif isinstance(v, str):
            if v:
                c = f'{c} --{k} {v}'
        else:
            c = f'{c} --{k} {v}'
    return c


def device_next_id() -> int:
    return next(device_ids_cycle_g)


parser = argparse.ArgumentParser(
    description="Experiment Grid Search Script"
)
parser.add_argument(
    'config', help="Experiments grid search configuration file"
)
parser.add_argument(
    '--dry-run', action='store_true', help="Whether to just print the commands without executing"
)
parser.add_argument(
    '--num-jobs', type=int, default=1, help="The number of processes to run in parallel (on a single device)"
)
parser.add_argument(
    '--multi-devices', type=str, default="",
    help="The list of device IDs to run in parallel, as an alternative to --n-jobs"
)
parser.add_argument(
    '--num-repetitions', type=int, default=1, help="The number of independent repetitions"
)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as fp:
        config = json.load(fp)
    dry_run = args.dry_run
    num_jobs = args.num_jobs
    multi_devices = args.multi_devices.split()
    assert num_jobs > 0
    if not multi_devices:
        assert num_jobs == 1, "Multiple jobs on multiple devices are not supported yet"
    else:
        device_ids_cycle_g = itertools.cycle(multi_devices)
    assert args.num_repetitions > 0

    common_hparams = config['common']
    common_hparams_grid = config['grid']['common']

    # Produce the list of commands
    commands = list()
    for dataset in config['datasets']:
        # Get the hyperparameters grid, based on the dataset
        hparams_grid_datasets = config['grid']['models'].keys()
        selected_ds = next(filter(lambda d: dataset in d.split('|'), hparams_grid_datasets))
        hparams_grid = config['grid']['models'][selected_ds]

        for model in hparams_grid:
            # # Get the list of hyperparameters, based on the model
            hparams = model_hparams(hparams_grid[model], common_hparams_grid)

            # Run each experiment
            for exp_alias, hps in hparams:
                hp = hps.copy()
                if exp_alias is not None:
                    hp['exp-alias'] = exp_alias
                hp.update(common_hparams)
                cmd = build_command_string(dataset, model, hp)
                device = device_next_id() if multi_devices else common_hparams['device']
                if args.num_repetitions == 1:
                    commands.append((cmd, device))
                    continue
                for k in range(args.num_repetitions):
                    rep_seed = 123 + 42 * k
                    rep_cmd = f'{cmd} --seed {rep_seed}'
                    commands.append((rep_cmd, device))

    # Run the commands, if --dry-run is not specified
    if (num_jobs == 1 and not multi_devices) or dry_run:
        for cmd, device in commands:
            cmd = f'{cmd} --device {device}'
            print(cmd)
            if not dry_run:
                subprocess.run(cmd.split())
    elif multi_devices:
        def run_multi_commands(device_cmds: List[str], stdout: int = subprocess.DEVNULL):
            for cmd in device_cmds:
                subprocess.run(cmd.split(), stdout=stdout)
        num_devices = len(multi_devices)
        commands_per_device = defaultdict(list)
        for cmd, device in commands:
            commands_per_device[device].append(f'{cmd} --device {device}')
        with multiprocessing.Pool(num_devices) as pool:
            for device, device_cmds in commands_per_device.items():
                pool.apply_async(run_multi_commands, args=[device_cmds])
            pool.close()
            pool.join()
    else:
        with multiprocessing.Pool(num_jobs) as pool:
            for cmd, device in commands:
                cmd = f'{cmd} --device {device}'
                pool.apply_async(
                    subprocess.run, args=[cmd.split()],
                    kwds=dict(stdout=subprocess.DEVNULL)
                )
            pool.close()
            pool.join()
