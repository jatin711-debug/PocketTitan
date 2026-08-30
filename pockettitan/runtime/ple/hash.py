"""Bit-exact n-gram hash calculator matching reference Transformers PLE block (R5)."""

from typing import List, Sequence, Union
import torch
from pockettitan.package.format import PleIndex


def signed_int64(value: int) -> int:
    """Simulate 64-bit signed integer arithmetic wraparound."""
    value &= 0xFFFFFFFFFFFFFFFF
    return value - (1 << 64) if value >= (1 << 63) else value


class PleHasher:
    """Resolves token sequences to exact row offsets in the PLE table."""

    def __init__(self, index: PleIndex):
        self.index = index
        self.ngram_size = index.ngram_size
        self.num_heads = index.num_heads
        self.heads_per_ngram = index.heads_per_ngram
        self.multipliers = index.layer_multipliers
        self.offsets = index.head_offsets
        self.vocab_sizes = index.head_vocab_sizes

    def hash_single_head(self, head: int, tokens: Sequence[int]) -> int:
        """Compute the logical row ID for a single head given recent token history."""
        return self.index.row_id(head, list(tokens))

    def hash_all_heads(self, tokens: Sequence[int]) -> List[int]:
        """Compute all 16 head row IDs for the current token step.
        
        Requires at least ngram_size (e.g. 3) tokens in history.
        """
        if len(tokens) < self.ngram_size:
            # Pad with zeros if prompt context has fewer tokens than ngram_size
            pad_len = self.ngram_size - len(tokens)
            tokens = [0] * pad_len + list(tokens)
            
        rows: List[int] = []
        for h in range(self.num_heads):
            rows.append(self.index.row_id(h, list(tokens)))
        return rows

    def hash_sequence_batched(self, token_ids: Sequence[int]) -> List[List[int]]:
        """Compute all 16 PLE head rows for every token position in a sequence (for prefill)."""
        seq_rows: List[List[int]] = []
        tok_list = list(token_ids)
        
        for i in range(len(tok_list)):
            history = tok_list[: i + 1]
            seq_rows.append(self.hash_all_heads(history))
            
        return seq_rows
