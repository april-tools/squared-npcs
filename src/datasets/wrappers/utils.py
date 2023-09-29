from typing import Callable, Tuple, Iterable

import torch

from torch.utils.data import IterableDataset

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab


def get_tokenizer_vocab(
        train_iter: Iterable,
        language: str = 'basic_english'
) -> Tuple[Callable, Vocab]:
    tokenizer = get_tokenizer(language)
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    return tokenizer, vocab


def process_text(raw_text_iter: IterableDataset, tokenizer: Callable, vocab: Vocab) -> torch.Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify_text(data: torch.Tensor, seq_length: int) -> torch.Tensor:
    num_seqs = data.shape[0] // seq_length
    data = data[:num_seqs * seq_length]
    return data.view(num_seqs, seq_length).contiguous()
