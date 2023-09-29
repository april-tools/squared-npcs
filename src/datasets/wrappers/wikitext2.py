from typing import Tuple

import torch

from torchtext.datasets import WikiText2

from datasets.wrappers.utils import get_tokenizer_vocab, process_text, batchify_text


def load_wikitext2(path: str = '.', seq_length: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_iter = WikiText2(root=path, split='train')
    tokenizer, vocab = get_tokenizer_vocab(train_iter)

    train_iter, val_iter, test_iter = WikiText2(root=path)
    train_data = process_text(train_iter, tokenizer, vocab)
    val_data = process_text(val_iter, tokenizer, vocab)
    test_data = process_text(test_iter, tokenizer, vocab)

    train_data = batchify_text(train_data, seq_length=seq_length)
    val_data = batchify_text(val_data, seq_length=seq_length)
    test_data = batchify_text(test_data, seq_length=seq_length)

    return train_data, val_data, test_data
