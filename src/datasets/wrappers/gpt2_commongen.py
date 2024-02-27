import os
import glob
from typing import Tuple

import numpy as np
import pandas as pd
import torch


def load_gpt2_commongen(
        path: str = 'datasets',
        seed: int = 42,
        num_splits: int = 40
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    splits = glob.glob(os.path.join(path, 'gpt2_commongen', 'common-gen.train.*')) 
    assert len(splits) > 0, f"There are no GPT2 CommonGen-tuned generated splits: {path} {splits}"
    assert len(splits) >= num_splits, f"There are no enough GPT2 CommonGen-tuned generated splits: {path} {splits}"
    splits = splits[:num_splits]
    samples = np.concatenate([pd.read_csv(f, header=None).to_numpy(dtype=np.int64) for f in splits], axis=0)

    random_state = np.random.RandomState(seed)
    random_state.shuffle(samples)
    num_valid, num_test = int(0.05 * len(samples)), int(0.1 * len(samples))
    num_train = len(samples) - (num_valid + num_test)
    train_data = torch.from_numpy(samples[:num_train])
    valid_data = torch.from_numpy(samples[num_train:num_train + num_valid])
    test_data = torch.from_numpy(samples[num_train + num_valid:])
    return train_data, valid_data, test_data
