"""Text datasets for language modeling and associative recall.

Two tasks that test the hybrid attention's sequential memory capabilities:

1. Tiny Shakespeare  — Next-character prediction on ~1MB of Shakespeare's plays.
   Tests whether the global memory + forget gate can learn real linguistic
   structure (grammar, word boundaries, sentence flow).

2. Associative Recall — Randomly generated key-value pairs followed by a query.
   Tests exact retrieval from the d×d memory matrix. Natural language can hide
   flaws (because it's predictable); random data cannot.

The core engine (hybrid_attention.py) is NOT touched. Only the input/output
wrappers change.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


# ── Character-level Tokenizer ────────────────────────────────────────────────

class CharTokenizer:
    """Simple character-level tokenizer.

    Maps each unique character to an integer ID.
    Typical vocab size for Shakespeare: ~65 characters.
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)


# ── Tiny Shakespeare Dataset ────────────────────────────────────────────────

_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _download_shakespeare() -> str:
    """Download Tiny Shakespeare (~1MB) if not already cached."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = _DATA_DIR / "tiny_shakespeare.txt"

    if not filepath.exists():
        urllib.request.urlretrieve(_SHAKESPEARE_URL, filepath)

    return filepath.read_text(encoding="utf-8")


class TinyShakespeareDataset(Dataset):
    """Next-character prediction dataset from Shakespeare's complete works.

    Each sample is a pair (input_ids, target_ids) where:
        input_ids  = characters[i : i + seq_len]
        target_ids = characters[i+1 : i + seq_len + 1]

    The model predicts each next character given the preceding context.

    Parameters
    ----------
    text : str
        The full text corpus.
    tokenizer : CharTokenizer
        Character-level tokenizer built from the same text.
    seq_len : int
        Context window length (number of characters per sample).
    """

    def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int = 128) -> None:
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[index : index + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ── Associative Recall Dataset ───────────────────────────────────────────────

class AssociativeRecallDataset(Dataset):
    """Synthetic associative recall benchmark.

    Each sample contains:
        1. A sequence of unique key-value pairs: [k1, v1, k2, v2, ...]
        2. A separator token
        3. A query key (one of the keys from the sequence)

    The target is the value that was paired with the query key.

    Example with num_pairs=4, vocab_size=100:
        Input:  [3, 17, 14, 82, 99, 56, 42, 9, SEP, 99]
        Target: 56  (the value paired with key 99)

    Every key appears exactly ONCE per sample — no ambiguity.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    num_pairs : int
        Number of key-value pairs per sequence.
    vocab_size : int
        Size of the token vocabulary (keys and values drawn from 0..vocab_size-1).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 5000,
        num_pairs: int = 8,
        vocab_size: int = 100,
        seed: int = 7,
    ) -> None:
        self.num_pairs = num_pairs
        self.vocab_size = vocab_size
        # Reserve two special tokens: SEP and PAD
        self.sep_token = vocab_size
        self.pad_token = vocab_size + 1
        self.total_vocab = vocab_size + 2  # include SEP and PAD

        generator = torch.Generator().manual_seed(seed)
        self.samples: list[tuple[torch.Tensor, int]] = []

        for _ in range(num_samples):
            # Pick num_pairs unique keys (no duplicates within one sample)
            perm = torch.randperm(vocab_size, generator=generator)
            keys = perm[:num_pairs]

            # Random values for each key
            values = torch.randint(0, vocab_size, (num_pairs,), generator=generator)

            # Build the sequence: [k1, v1, k2, v2, ..., SEP, query_key]
            seq = torch.zeros(num_pairs * 2 + 2, dtype=torch.long)
            for i in range(num_pairs):
                seq[i * 2] = keys[i]
                seq[i * 2 + 1] = values[i]

            # Pick a random query key
            query_idx = torch.randint(0, num_pairs, (1,), generator=generator).item()
            seq[-2] = self.sep_token
            seq[-1] = keys[query_idx]

            # Target = the value paired with the query key
            target = values[query_idx].item()

            self.samples.append((seq, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, target = self.samples[index]
        return seq, torch.tensor(target, dtype=torch.long)

    @property
    def seq_len(self) -> int:
        """Length of each input sequence."""
        return self.num_pairs * 2 + 2


# ── Bundle & Factory ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TextDataBundle:
    """Container for text task data, analogous to DatasetBundle for images."""
    train: DataLoader
    val: DataLoader
    vocab_size: int
    seq_len: int
    task_type: str  # "shakespeare" or "recall"
    tokenizer: CharTokenizer | None = None  # only for shakespeare


def make_text_dataset(
    name: str,
    batch_size: int,
    seed: int,
    seq_len: int = 128,
    limit: int = 0,
    num_pairs: int = 8,
    recall_vocab: int = 100,
) -> TextDataBundle:
    """Factory for text datasets.

    Parameters
    ----------
    name : str
        "shakespeare" or "recall"
    batch_size : int
    seed : int
    seq_len : int
        Context window for Shakespeare (ignored for recall).
    limit : int
        Max training samples (0 = use all available).
    num_pairs : int
        Key-value pairs per sample (recall only).
    recall_vocab : int
        Vocabulary size for recall task.
    """
    name = name.lower()

    if name == "shakespeare":
        text = _download_shakespeare()
        tokenizer = CharTokenizer(text)

        # Split: 90% train, 10% val
        split_idx = int(len(text) * 0.9)
        train_text = text[:split_idx]
        val_text = text[split_idx:]

        train_ds = TinyShakespeareDataset(train_text, tokenizer, seq_len)
        val_ds = TinyShakespeareDataset(val_text, tokenizer, seq_len)

        # Apply sample limit if requested
        if limit > 0 and limit < len(train_ds):
            train_ds = torch.utils.data.Subset(train_ds, range(limit))

        return TextDataBundle(
            train=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            val=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            vocab_size=tokenizer.vocab_size,
            seq_len=seq_len,
            task_type="shakespeare",
            tokenizer=tokenizer,
        )

    elif name == "recall":
        train_count = limit if limit > 0 else 5000
        val_count = max(200, train_count // 5)

        train_ds = AssociativeRecallDataset(
            num_samples=train_count,
            num_pairs=num_pairs,
            vocab_size=recall_vocab,
            seed=seed,
        )
        val_ds = AssociativeRecallDataset(
            num_samples=val_count,
            num_pairs=num_pairs,
            vocab_size=recall_vocab,
            seed=seed + 1,
        )

        return TextDataBundle(
            train=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            val=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            vocab_size=train_ds.total_vocab,
            seq_len=train_ds.seq_len,
            task_type="recall",
        )

    else:
        raise ValueError(f"Unknown text dataset: {name}. Use 'shakespeare' or 'recall'.")
