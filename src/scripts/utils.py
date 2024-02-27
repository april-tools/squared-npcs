import os
import random
import subprocess
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import torch
from zuko.flows import Flow, NICE, MAF, NSF
from sklearn.preprocessing import StandardScaler
from tbparse import SummaryReader
from torch.utils.data import DataLoader, TensorDataset
import wandb

from pcs.hmm import MonotonicHMM, BornHMM
from pcs.utils import retrieve_default_dtype
from datasets.loaders import BINARY_DATASETS, IMAGE_DATASETS, CONTINUOUS_DATASETS, SMALL_UCI_DATASETS, \
    load_small_uci_dataset, LANGUAGE_DATASETS, load_language_dataset
from datasets.loaders import load_image_dataset, load_binary_dataset, load_artificial_dataset, load_continuous_dataset
from pcs.layers import MonotonicEmbeddings, MonotonicBinaryEmbeddings, MultivariateNormalDistribution, \
    NormalDistribution, BornEmbeddings, BornBinaryEmbeddings, \
    BornMultivariateNormalDistribution, BornNormalDistribution, MonotonicBSplines, BornBSplines, \
    MonotonicBinomial, BornBinomial
from pcs.layers.mixture import MonotonicMixtureLayer, BornMixtureLayer
from pcs.layers.tucker import MonotonicTucker2Layer, BornTucker2Layer
from pcs.layers.candecomp import MonotonicCPLayer, BornCPLayer
from pcs.models import PC, MonotonicPC, BornPC
from graphics.distributions import plot_bivariate_samples_hmap, plot_bivariate_discrete_samples_hmap, kde_samples_hmap
from region_graph import RegionGraph, RegionNode
from region_graph.linear_vtree import LinearVTree
from region_graph.quad_tree import QuadTree
from region_graph.random_binary_tree import RandomBinaryTree

WANDB_KEY_FILE = "wandb_api.key" # Put your wandb api key in this file, first line

def drop_na(df: pd.DataFrame, drop_cols: List[str], verbose: bool=True) -> pd.DataFrame:
    N = len(df)
        
    for c in drop_cols:
        if verbose:
            print(f"Dropping {len(df[pd.isna(df[c])])} runs that do not contain values for {c}")
        df = df[~pd.isna(df[c])]

    if verbose:
        print(f"Dropped {N - len(df)} out of {N} rows.")

    return df



def filter_dataframe(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in filter_dict.items():
        # If v is a list, filter out rows with values NOT in the list
        if isinstance(v, list):
            df = df[df[k].isin(v)]
        else:
            if isinstance(v, bool):
                v = float(v)
            df = df[df[k] == v]
    return df


def unroll_hparams(hparams: dict) -> List[dict]:
    """
    :param hparams: dictionary with hyperparameter names as keys and hyperparam value domain as list

    Returns 
    """
    unroll_hparams = [dict()]
    for k in hparams:
        vs = hparams[k]
        new_unroll_hparams = list()
        for v in vs:
            for hp in unroll_hparams:
                new_hp = hp.copy()
                new_hp[k] = v
                new_unroll_hparams.append(new_hp)
        unroll_hparams = new_unroll_hparams
    return unroll_hparams


def format_model(m: str, exp_reparam: bool = False) -> str:
    if m == 'MonotonicPC':
        return r"$+$"
    elif m == 'BornPC':
        if exp_reparam:
            return r"$+^2$"
        else:
            return r"$\pm^2$"
    assert False


def set_global_seed(seed: int, is_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8')


def retrieve_wandb_runs(project_names: Union[str, List[str]], verbose: bool=True) -> pd.DataFrame:
    """
    Returns all wandb runs from project name(s) specified

    :param project_names: The wandb user or team name and the project name as a string e.g "user12/project34" 
    :param verbose: Bool for printing messages about processing
    """
    api = wandb.Api(api_key=open(WANDB_KEY_FILE, "r").readline())

    if isinstance(project_names, str):
        project_names = [project_names]

    # Project is specified by <entity/project-name>
    runs = []
    for project_name in project_names:
        runs += api.runs(project_name)

    if verbose:
        print(f"Loaded {len(runs)} from wandb project(s): {','.join(project_names)}")

    #summary_list, config_list, name_list = [], [], []
    run_dicts = []
    for run in runs:
        run_dict = dict()
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        #summary_list.append(run.summary._json_dict)
        run_dict.update(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config = {k: v for k,v in run.config.items() if not k.startswith('_')}
        #config_list.append(config)
        
        run_dict.update(config)

        # .name is the human-readable name of the run.
        #name_list.append(run.name)
        run_dict.update({"name": run.name})
        run_dicts.append(run_dict)

    runs_df = pd.DataFrame(run_dicts)

    return runs_df


def retrieve_tboard_runs(tboard_path: str, metrics: Union[str, List[str]], ignore_diverged=False) -> pd.DataFrame:
    reader = SummaryReader(tboard_path, pivot=True, extra_columns={'dir_name'})
    df_hparams = reader.hparams
    df_scalars = reader.scalars

    if not isinstance(metrics, list):
        metrics = [metrics]
    
    print(f"N experiments: {len(df_hparams)}")
    # Throw out rows with no result for the metric
    for m in metrics:
        df_scalars = df_scalars[~pd.isna(df_scalars[m])]

    assert len(df_hparams) == len(df_scalars), "Number of runs and results is different"
    if ignore_diverged:
        n_diverged = int(np.sum(df_scalars['diverged']))
        print(f"Found {n_diverged} diverged runs. Ignoring...")
        df_scalars = df_scalars[df_scalars['diverged'] == False]
    df = df_hparams.merge(df_scalars, on='dir_name', sort=True).drop('dir_name', axis=1)

    return df


def retrieve_tboard_df(tboard_path: str) -> pd.DataFrame:
    reader = SummaryReader(tboard_path, pivot=True, extra_columns={'dir_name'})
    df_hparams = reader.hparams
    df_scalars = reader.scalars

    df_scalars = df_scalars[~pd.isna(df_scalars["Best/Test/avg_ll"])]

    # df_scalars = df_scalars.dropna(axis=1).drop('step', axis=1)
    df = df_hparams.merge(df_scalars, on='dir_name', sort=True).drop('dir_name', axis=1)

    print(len(df_hparams))
    return df


def retrieve_tboard_images(tboard_path: str) -> pd.DataFrame:
    reader = SummaryReader(tboard_path, pivot=False, extra_columns={'dir_name'})
    df_images = reader.images
    return df_images


def bits_per_dimension(average_ll: float, num_variables: int) -> float:
    return -average_ll / (num_variables * np.log(2.0))


def perplexity(average_ll: float, num_variables: int) -> float:
    return np.exp(-average_ll / num_variables)


def build_run_id(args):
    rs = list()
    if 'PC' in args.model:
        rs.append(f"RG{args.region_graph[:3]}")
        rs.append(f"R{args.num_replicas}")
    rs.append(f"K{args.num_components}")
    if 'PC' in args.model:
        rs.append(f"D{args.depth}")
        rs.append(f"L{args.compute_layer[:2]}")
    rs.append(f"O{args.optimizer}")
    rs.append(f"LR{args.learning_rate}")
    rs.append(f"BS{args.batch_size}")
    if 'PC' in args.model:
        if args.splines:
            rs.append(f"SO{args.spline_order}_SK{args.spline_knots}")
        if args.exp_reparam:
            rs.append(f"RExp")
    if 'PC' in args.model or 'HMM' in args.model:
        init_method_id = ''.join(m[0].upper() for m in args.init_method.split('-'))
        rs.append(f"I{init_method_id}")
    if args.weight_decay > 0.0:
        rs.append(f"WD{args.weight_decay}")
    return '_'.join(rs)


@torch.no_grad()
def evaluate_model_log_likelihood(
        model: Union[PC, Flow],
        dataloader: DataLoader,
        device: torch.device
) -> Tuple[float, float]:
    model.eval()
    lls = list()
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        batch = batch.to(device)
        if isinstance(model, PC):
            log_probs = model.log_prob(batch)
        else:
            log_probs = model().log_prob(batch)
        if len(log_probs.shape) > 1:
            log_probs.squeeze(dim=1)
        lls.extend(log_probs.tolist())
    return np.mean(lls).item(), np.std(lls).item()


def setup_experiment_path(root: str, dataset: str, model_name: str, alias: str = '', trial_id: str = ''):
    return os.path.join(root, dataset, model_name, alias, trial_id)


def setup_data_loaders(
        dataset: str,
        path: str,
        batch_size: int,
        num_workers: int = 0,
        num_samples: int = 1000,
        standardize: bool = False,
        dequantize: bool = False,
        discretize: bool = False,
        discretize_unique: bool = False,
        discretize_bins: int = 32,
        shuffle_bins: bool = False
) -> Tuple[dict, Tuple[DataLoader, DataLoader, DataLoader]]:
    seed = 123
    numpy_dtype = retrieve_default_dtype(numpy=True)
    metadata = dict()
    # Load the dataset
    small_uci_dataset = dataset in SMALL_UCI_DATASETS
    binary_dataset = dataset in BINARY_DATASETS
    image_dataset = dataset in IMAGE_DATASETS
    continuous_dataset = dataset in CONTINUOUS_DATASETS
    language_dataset = dataset in LANGUAGE_DATASETS
    if small_uci_dataset:
        train_data, valid_data, test_data = load_small_uci_dataset(dataset, path=path, seed=seed)
        metadata['image_shape'] = None
        metadata['num_variables'] = train_data.shape[1]
        metadata['hmap'] = None
        metadata['type'] = 'categorical'
        max_state_value = max(np.max(train_data), np.max(valid_data), np.max(test_data))
        metadata['interval'] = (0, max_state_value)
        metadata['domains'] = None
    elif image_dataset:
        image_shape, (train_data, valid_data, test_data), (train_label, valid_label, test_label) = load_image_dataset(
            dataset, path=path, dequantize=dequantize, dtype=numpy_dtype)
        metadata['image_shape'] = image_shape
        metadata['num_variables'] = np.prod(image_shape).item()
        metadata['hmap'] = None
        metadata['type'] = 'image'
        if dequantize:
            logit_eps = 1e-6
            limits = np.array([0.0, 1.0], dtype=numpy_dtype)
            limits = logit_eps + (1.0 - 2.0 * logit_eps) * limits
            limits = np.log(limits) - np.log(1.0 - limits)
            metadata['interval'] = (limits[0], limits[1])
        else:
            metadata['interval'] = (0, 255)
        metadata['domains'] = None
        train_data = TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
        valid_data = TensorDataset(torch.tensor(valid_data), torch.tensor(valid_label))
        test_data = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
    elif binary_dataset:
        sep = ','
        if dataset == 'binarized_mnist':
            sep = ' '
        train_data, valid_data, test_data = load_binary_dataset(dataset, path=path, sep=sep)
        metadata['num_variables'] = train_data.shape[1]
        metadata['hmap'] = None
        metadata['domains'] = None
        if dataset == 'binarized_mnist':
            metadata['image_shape'] = (1, 28, 28)
            metadata['type'] = 'image'
            metadata['interval'] = (0, 1)
        else:
            metadata['image_shape'] = None
            metadata['type'] = 'binary'
            metadata['interval'] = (0, 1)
    elif continuous_dataset:
        train_data, valid_data, test_data = load_continuous_dataset(
            dataset, path=path, dtype=numpy_dtype
        )
        train_valid_data = np.concatenate([train_data, valid_data], axis=0)
        data_min = np.min(train_valid_data, axis=0)
        data_max = np.max(train_valid_data, axis=0)
        drange = np.abs(data_max - data_min)
        data_min, data_max = (data_min - drange * 0.05), (data_max + drange * 0.05)
        metadata['image_shape'] = None
        metadata['num_variables'] = train_data.shape[1]
        metadata['hmap'] = None
        metadata['type'] = 'continuous'
        metadata['interval'] = (np.min(data_min), np.max(data_max))
        metadata['domains'] = [(data_min[i], data_max[i]) for i in range(len(data_min))]
        train_data = TensorDataset(torch.tensor(train_data))
        valid_data = TensorDataset(torch.tensor(valid_data))
        test_data = TensorDataset(torch.tensor(test_data))
    elif language_dataset:
        train_data, valid_data, test_data = load_language_dataset(dataset, path=path, seed=seed)
        seq_length = train_data.shape[1]
        metadata['image_shape'] = None
        metadata['num_variables'] = seq_length
        metadata['hmap'] = None
        metadata['type'] = 'language'
        metadata['interval'] = (torch.min(train_data).item(), torch.max(train_data).item())
        metadata['domains'] = None
        train_data = TensorDataset(train_data)
        valid_data = TensorDataset(valid_data)
        test_data = TensorDataset(test_data)
    else:
        train_data, valid_data, test_data = load_artificial_dataset(
            dataset, num_samples=num_samples, discretize=discretize, discretize_unique=discretize_unique,
            discretize_bins=discretize_bins, shuffle_bins=shuffle_bins, dtype=retrieve_default_dtype(numpy=True)
        )
        metadata['image_shape'] = None
        metadata['num_variables'] = 2
        if discretize:
            metadata['type'] = 'categorical'
            metadata['interval'] = (0, discretize_bins - 1)
            metadata['domains'] = [(0, discretize_bins - 1), (0, discretize_bins - 1)]
            metadata['hmap'] = plot_bivariate_discrete_samples_hmap(
                train_data, xlim=metadata['domains'][0], ylim=metadata['domains'][1])
        else:
            if standardize:
                scaler = StandardScaler()
                scaler.fit(train_data)
                train_data = scaler.transform(train_data)
                valid_data = scaler.transform(valid_data)
                test_data = scaler.transform(test_data)
            train_valid_data = np.concatenate([train_data, valid_data], axis=0)
            data_min = np.min(train_valid_data, axis=0)
            data_max = np.max(train_valid_data, axis=0)
            drange = np.abs(data_max - data_min)
            data_min, data_max = (data_min - drange * 0.05), (data_max + drange * 0.05)
            metadata['type'] = 'artificial'
            metadata['interval'] = (np.min(data_min), np.max(data_max))
            metadata['domains'] = [(data_min[i], data_max[i]) for i in range(len(data_min))]
            metadata['hmap'] = kde_samples_hmap(train_data, xlim=metadata['domains'][0], ylim=metadata['domains'][1])
    train_dataloader = DataLoader(train_data, batch_size, num_workers=num_workers, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size, num_workers=num_workers)
    return metadata, (train_dataloader, valid_dataloader, test_dataloader)


def setup_model(
        model_name: str,
        dataset_metadata: dict,
        rg_type: str = 'random',
        rg_replicas: int = 1,
        rg_depth: int = 1,
        num_components: int = 2,
        input_mixture: bool = False,
        compute_layer: str = 'cp',
        multivariate: bool = False,
        binomials: bool = False,
        splines: bool = False,
        spline_order: int = 2,
        spline_knots: int = 8,
        exp_reparam: bool = False,
        init_method: str = 'normal',
        init_scale: float = 1.0,
        dequantize: bool = False,
        l2norm: bool = False,
        seed: int = 123
) -> Union[PC, Flow]:
    if binomials and splines:
        raise ValueError("At most one between --binomials and --splines must be true")
    if exp_reparam and model_name != 'BornPC':
        raise ValueError("--exp-reparam can only be used with BornPC models")
    dataset_type = dataset_metadata['type']
    num_variables = dataset_metadata['num_variables']
    image_shape = dataset_metadata['image_shape']
    interval = dataset_metadata['interval']
    input_layer_kwargs = dict()
    if model_name == 'MonotonicPC':
        if dataset_type == 'image':
            if splines:
                input_layer_cls = MonotonicBSplines
                input_layer_kwargs['order'] = spline_order
                input_layer_kwargs['num_knots'] = spline_knots
                input_layer_kwargs['interval'] = interval
            elif dequantize:
                input_layer_cls = NormalDistribution
            else:
                input_layer_cls = MonotonicBinomial if binomials else MonotonicEmbeddings
                input_layer_kwargs['num_states'] = interval[1] + 1
        elif dataset_type in ['categorical', 'language']:
            input_layer_cls = MonotonicBinomial if binomials else MonotonicEmbeddings
            input_layer_kwargs['num_states'] = interval[1] + 1
        elif dataset_type == 'binary':
            input_layer_cls = MonotonicBinaryEmbeddings
        else:
            if splines:
                input_layer_cls = MonotonicBSplines
                input_layer_kwargs['order'] = spline_order
                input_layer_kwargs['num_knots'] = spline_knots
                input_layer_kwargs['interval'] = interval
            else:
                input_layer_cls = MultivariateNormalDistribution if multivariate else NormalDistribution
        model_cls = MonotonicPC
        compute_layer_cls = MonotonicCPLayer if compute_layer == 'cp' else MonotonicTucker2Layer
        out_mixture_layer_cls = MonotonicMixtureLayer
        in_mixture_layer_cls = MonotonicMixtureLayer
    elif model_name == 'BornPC':
        if dataset_type == 'image':
            if splines:
                input_layer_cls = BornBSplines
                input_layer_kwargs['order'] = spline_order
                input_layer_kwargs['num_knots'] = spline_knots
                input_layer_kwargs['interval'] = interval
            elif dequantize:
                input_layer_cls = BornNormalDistribution
            else:
                input_layer_cls = BornBinomial if binomials else BornEmbeddings
                input_layer_kwargs['num_states'] = interval[1] + 1
            out_mixture_layer_cls = MonotonicMixtureLayer
        elif dataset_type in ['categorical', 'language']:
            input_layer_cls = BornBinomial if binomials else BornEmbeddings
            input_layer_kwargs['num_states'] = interval[1] + 1
            out_mixture_layer_cls = MonotonicMixtureLayer
        elif dataset_type == 'binary':
            input_layer_cls = BornBinaryEmbeddings
            out_mixture_layer_cls = MonotonicMixtureLayer
        else:
            if splines:
                input_layer_cls = BornBSplines
                input_layer_kwargs['order'] = spline_order
                input_layer_kwargs['num_knots'] = spline_knots
                input_layer_kwargs['interval'] = interval
            else:
                input_layer_cls = BornMultivariateNormalDistribution if multivariate else BornNormalDistribution
            out_mixture_layer_cls = BornMixtureLayer if multivariate else MonotonicMixtureLayer
        model_cls = BornPC
        compute_layer_cls = BornCPLayer if compute_layer == 'cp' else BornTucker2Layer
        in_mixture_layer_cls = BornMixtureLayer
    elif 'HMM' in model_name:
        model_cls = MonotonicHMM if 'Monotonic' in model_name else BornHMM
        kwargs = dict() if 'Monotonic' in model_name else {'l2norm': l2norm}
        assert dataset_type == 'language'
        model = model_cls(
            vocab_size=interval[1] + 1,
            seq_length=num_variables,
            hidden_size=num_components,
            init_method=init_method,
            init_scale=init_scale,
            **kwargs
        )
        return model
    elif model_name == 'NICE':
        if dataset_type not in ['continuous', 'artificial']:
            raise ValueError("NICE is not supported for the requested data set")
        model = NICE(
            features=num_variables,
            transforms=10,
            hidden_features=(num_components, num_variables)
        )
        return model
    elif model_name == 'MAF':
        if dataset_type not in ['continuous', 'artificial']:
            raise ValueError("MAF is not supported for the requested data set")
        model = MAF(
            features=num_variables,
            transforms=10,
            hidden_features=(num_components, num_variables)
        )
        return model
    elif model_name == 'NSF':
        if dataset_type not in ['continuous', 'artificial']:
            raise ValueError("MAF is not supported for the requested data set")
        model = NSF(
            features=num_variables,
            transforms=10,
            hidden_features=(num_components, num_variables),
            bins=8
        )
        return model
    else:
        raise ValueError(f"Unknown model called {model_name}")
    # Instantiate the region graph and the model
    if dataset_type in ['image', 'binary', 'continuous']:
        if dataset_type == 'continuous' and multivariate:
            rg = RegionGraph()
            rg.add_node(RegionNode(range(num_variables)))
        elif rg_type == 'random':
            rg = RandomBinaryTree(
                num_variables, num_repetitions=rg_replicas, depth=rg_depth, seed=seed)
        elif rg_type == 'quad-tree' and dataset_type == 'image':
            rg = QuadTree(image_shape, struct_decomp=True)
        elif rg_type == 'linear-vtree':
            rg = LinearVTree(num_variables, num_repetitions=rg_replicas, randomize=True, seed=seed)
        else:
            raise ValueError(f"Unknown region graph type named {rg_type} for the selected data")
    elif dataset_type in ['language']:
        if rg_type == 'linear-vtree':
            rg = LinearVTree(num_variables, num_repetitions=rg_replicas, randomize=False, seed=seed)
        else:
            raise ValueError(f"Unknown region graph type named {rg_type} for the selected data")
    else:
        if multivariate:
            rg = RegionGraph()
            rg.add_node(RegionNode(range(num_variables)))
        elif rg_type == 'random':
            rg = RandomBinaryTree(
                num_variables, num_repetitions=rg_replicas, depth=rg_depth, seed=seed)
        elif rg_type == 'linear-vtree':
            rg = LinearVTree(num_variables, num_repetitions=rg_replicas, randomize=True, seed=seed)
        else:
            raise ValueError(f"Unknown region graph type named {rg_type} for the selected data")
    if all(n not in input_layer_cls.__name__ for n in ['Normal', 'Binomial']):
        input_layer_kwargs['init_method'] = init_method
    if all(n not in input_layer_cls.__name__ for n in ['Binomial']):
        input_layer_kwargs['init_scale'] = init_scale
    compute_layer_kwargs = {
        'init_method': init_method,
        'init_scale': init_scale
    }
    if model_name == 'BornPC':
        if all(n not in input_layer_cls.__name__ for n in ['Normal', 'Binomial']):
            input_layer_kwargs['exp_reparam'] = exp_reparam
        if 'Embeddings' in input_layer_cls.__name__:
            input_layer_kwargs['l2norm'] = l2norm
        compute_layer_kwargs['exp_reparam'] = exp_reparam
    return model_cls(
        rg,
        input_layer_cls=input_layer_cls,
        compute_layer_cls=compute_layer_cls,
        out_mixture_layer_cls=out_mixture_layer_cls,
        in_mixture_layer_cls=in_mixture_layer_cls,
        num_components=num_components,
        input_mixture=input_mixture,
        input_layer_kwargs=input_layer_kwargs,
        compute_layer_kwargs=compute_layer_kwargs,
        dequantize=dequantize)
