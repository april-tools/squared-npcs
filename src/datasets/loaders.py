import csv
import os
import pickle
import time
from typing import Optional, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from datasets.wrappers import *
from datasets.wrappers.artificial import single_ring_sample, multi_rings_sample, funnel_sample, banana_sample, \
    cosine_sample, spiral_sample, rotate_samples
from datasets.wrappers.gpt2_commongen import load_gpt2_commongen

SMALL_UCI_DATASETS = [
    "biofam",
    "flare",
    "lymphography",
    "spect",
    "tumor",
    "votes"
]

BINARY_DATASETS = [
    "accidents",
    "ad",
    "baudio",
    "bbc",
    "binarized_mnist",
    "bnetflix",
    "book",
    "c20ng",
    "cr52",
    "cwebkb",
    "dna",
    "jester",
    "kdd",
    "kosarek",
    "msnbc",
    "msweb",
    "mushrooms",
    "nltcs",
    "ocr_letters",
    "plants",
    "pumsb_star",
    "tmovie",
    "tretail"
]

IMAGE_DATASETS = [
    "MNIST",
    "FashionMNIST",
    "EMNIST",
    "CIFAR10"
]

CONTINUOUS_DATASETS = [
    'power',
    'gas',
    'hepmass',
    'miniboone',
    'bsds300'
]

ARTIFICIAL_DATASETS = [
    "ring",
    "mring",
    "funnel",
    "banana",
    "cosine",
    "spiral"
]

LANGUAGE_DATASETS = [
    "wikitext2",
    "wikitext103",
    "gpt2_commongen"
]


ALL_DATASETS = SMALL_UCI_DATASETS +\
               BINARY_DATASETS +\
               IMAGE_DATASETS +\
               CONTINUOUS_DATASETS +\
               ARTIFICIAL_DATASETS +\
               LANGUAGE_DATASETS


def load_small_uci_dataset(
    name: str,
    path: str = 'datasets',
    dtype: str = 'int64',
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split small UCI datasets.
    """
    with open(os.path.join(path, name), 'rb') as f:
        data = pickle.load(f, encoding='latin1')[0]
        data = data.astype(dtype, copy=False)
    random_state = np.random.RandomState(seed)
    data = random_state.permutation(data)
    unique_values = np.unique(data)
    assert np.all(unique_values == np.arange(len(unique_values)))
    num_samples = data.shape[0]
    num_unseen_samples = max(1, int(0.05 * num_samples))
    num_train_samples = num_samples - 2 * num_unseen_samples
    train_data = data[:num_train_samples]
    valid_data = data[num_train_samples:num_train_samples + num_unseen_samples]
    test_data = data[num_train_samples + num_unseen_samples:]
    return train_data, valid_data, test_data


def csv_2_numpy(filename: str, path: str = 'datasets', sep: str = ',', dtype: str = 'int8') -> np.ndarray:
    """
    Utility to read a dataset in csv format into a numpy array.
    """
    file_path = os.path.join(path, filename)
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    array = np.array(x, dtype=dtype)
    return array


def load_binary_dataset(
    name: str,
    path: str = 'datasets',
    sep: str = ',',
    dtype: str = 'int64',
    suffix: str = 'data',
    splits: Optional[List[str]] = None,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Loading training, validation and test splits by suffix from csv files.
    """
    if splits is None:
        splits = ["train", "valid", "test"]
    csv_files = [
        os.path.join(name, "{0}.{1}.{2}".format(name, ext, suffix)) for ext in splits
    ]

    load_start_t = time.perf_counter()
    dataset_splits = [csv_2_numpy(file, path, sep, dtype) for file in csv_files]
    load_end_t = time.perf_counter()

    if verbose:
        print("Dataset splits for {0} loaded in {1} secs".format(name, load_end_t - load_start_t))
        for data, split in zip(dataset_splits, splits):
            print("\t{0}:\t{1}".format(split, data.shape))
    return dataset_splits


def load_image_dataset(
        name: str, path: str = 'datasets', dequantize: bool = False, flip: bool = False, dtype: np.dtype = torch.float32
) -> Tuple[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if name == 'MNIST':
        data = MNIST(path, dequantize=dequantize)
    elif name == 'CIFAR10':
        data = CIFAR10(path, flip=flip, dequantize=dequantize)
    else:
        raise ValueError(f"Unknown datasets called {name}")
    image_train, image_valid, image_test = data.trn.x, data.val.x, data.tst.x
    label_train, label_valid, label_test = data.trn.y, data.val.y, data.tst.y
    image_shape = data.image_shape
    if dequantize:
        image_train = image_train.astype(dtype, copy=False)
        image_valid = image_valid.astype(dtype, copy=False)
        image_test = image_test.astype(dtype, copy=False)
    return image_shape, (image_train, image_valid, image_test), (label_train, label_valid, label_test)


def load_continuous_dataset(
    name: str,
    path: str = 'datasets',
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name == 'power':
        data = POWER(path)
        data_train, data_valid, data_test = data.trn.x, data.val.x, data.tst.x
    elif name == 'gas':
        data = GAS(path)
        data_train, data_valid, data_test = data.trn.x, data.val.x, data.tst.x
    elif name == 'hepmass':
        data = HEPMASS(path)
        data_train, data_valid, data_test = data.trn.x, data.val.x, data.tst.x
    elif name == 'miniboone':
        data = MINIBOONE(path)
        data_train, data_valid, data_test = data.trn.x, data.val.x, data.tst.x
    elif name == 'bsds300':
        data = BSDS300(path)
        data_train, data_valid, data_test = data.trn.x, data.val.x, data.tst.x
    else:
        raise ValueError(f"Unknown continuous dataset called {name}")

    data_train = data_train.astype(dtype, copy=False)
    data_valid = data_valid.astype(dtype, copy=False)
    data_test = data_test.astype(dtype, copy=False)
    return data_train, data_valid, data_test


def load_artificial_dataset(
    name: str,
    num_samples: int,
    valid_test_perc: float = 0.2,
    seed: int = 42,
    dtype: np.dtype = np.float32,
    discretize: bool = False,
    discretize_unique: bool = False,
    discretize_bins: int = 32,
    shuffle_bins: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_valid_samples = int(num_samples * valid_test_perc * 0.5)
    num_test_samples = int(num_samples * valid_test_perc)
    total_num_samples = num_samples + num_valid_samples + num_test_samples
    if name == 'ring':
        data = single_ring_sample(total_num_samples, seed=seed, **kwargs)
    elif name == 'mring':
        data = multi_rings_sample(total_num_samples, seed=seed, **kwargs)
    elif name == 'funnel':
        data = funnel_sample(total_num_samples, seed=seed, **kwargs)
        data = rotate_samples(data)
    elif name == 'banana':
        data = banana_sample(total_num_samples, seed=seed, **kwargs)
    elif name == 'cosine':
        data = cosine_sample(total_num_samples, seed=seed, **kwargs)
        data = rotate_samples(data)
    elif name == 'spiral':
        data = spiral_sample(total_num_samples, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown dataset called {name}")
    data = data.astype(dtype=dtype, copy=False)

    if discretize:
        # Standardize data before "quantizing" it
        data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-10)
        xlim, ylim = (np.min(data[:, 0]), np.max(data[:, 0])), (np.min(data[:, 1]), np.max(data[:, 1]))
        _, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=discretize_bins, range=[xlim, ylim])
        quantized_xdata = np.searchsorted(xedges[:-1], data[:, 0], side='right') - 1
        quantized_ydata = np.searchsorted(yedges[:-1], data[:, 1], side='right') - 1
        if shuffle_bins:
            perm_state = np.random.RandomState(seed)
            state_permutation = perm_state.permutation(discretize_bins)
            quantized_xdata = state_permutation[quantized_xdata]
            quantized_ydata = state_permutation[quantized_ydata]
        data = np.stack([quantized_xdata, quantized_ydata], axis=1)
        if discretize_unique:
            data = np.unique(data, axis=0)
            num_samples = len(data)
            valid_test_perc *= 0.5
            num_valid_samples = int(num_samples * valid_test_perc * 0.5)
            num_test_samples = int(num_samples * valid_test_perc)

    train_data, valid_test_data = train_test_split(
        data, test_size=num_valid_samples + num_test_samples, shuffle=True, random_state=seed)
    valid_data, test_data = train_test_split(
        valid_test_data, test_size=num_test_samples, shuffle=False)
    return train_data, valid_data, test_data


def load_language_dataset(
        name: str,
        path: str = 'datasets',
        seq_length: int = 32,
        seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = os.path.join(path, 'language')
    if name == 'wikitext2':
        train_data, valid_data, test_data = load_wikitext2(path=path, seq_length=seq_length)
    elif name == 'wikitext103':
        train_data, valid_data, test_data = load_wikitext103(path=path, seq_length=seq_length)
    elif name == 'gpt2_commongen':
        train_data, valid_data, test_data = load_gpt2_commongen(path=path, seed=seed)
    else:
        raise ValueError(f"Unknown language dataset called {name}")
    return train_data, valid_data, test_data
