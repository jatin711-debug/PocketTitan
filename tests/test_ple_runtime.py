"""Tests for PLE runtime hasher and SSD row store (R5)."""

import tempfile
from pathlib import Path
import pytest
import torch

from pockettitan.package.format import PleIndex, PleRowLayout
from pockettitan.runtime.ple import PleHasher, PleRowStore


@pytest.fixture
def dummy_ple_index() -> PleIndex:
    """Construct a minimal valid PleIndex for unit testing."""
    row_layout = PleRowLayout.build(
        row_width=160,
        bits=4.0,
        group_size=160,
        symmetric=True,
    )
    
    num_heads = 16
    vocab_sizes = [20000003 + i * 10 for i in range(num_heads)]
    offsets = [i * 100000 for i in range(num_heads)]
    multipliers = [23703573157769, 20109073645365, 8052911324071]
    
    return PleIndex(
        ngram_size=3,
        num_heads=num_heads,
        heads_per_ngram=8,
        head_offsets=offsets,
        head_vocab_sizes=vocab_sizes,
        layer_multipliers=multipliers,
        total_rows=1600000,
        physical_rows=1000,
        shards=[],
        row=row_layout,
        source_layer=1,
    )


def test_ple_hasher_matches_index_and_wraps(dummy_ple_index):
    """Verify PleHasher matches PleIndex and handles large token values cleanly."""
    hasher = PleHasher(dummy_ple_index)
    
    tokens = [248000, 199999, 150000]
    row_h0 = hasher.hash_single_head(0, tokens)
    expected_h0 = dummy_ple_index.row_id(0, tokens)
    assert row_h0 == expected_h0
    
    all_rows = hasher.hash_all_heads(tokens)
    assert len(all_rows) == 16
    for h in range(16):
        assert all_rows[h] == dummy_ple_index.row_id(h, tokens)


def test_ple_hasher_batched_prefill(dummy_ple_index):
    """Verify prefill batch hashing generates correct sequence lengths."""
    hasher = PleHasher(dummy_ple_index)
    seq = [10, 20, 30, 40, 50]
    batch_rows = hasher.hash_sequence_batched(seq)
    
    assert len(batch_rows) == 5
    for row_list in batch_rows:
        assert len(row_list) == 16


def test_ple_row_store_roundtrip(dummy_ple_index):
    """Verify PleRowStore writes, reads, page-aligns, and decodes binary table rows."""
    row_layout = dummy_ple_index.row
    total_physical_rows = 100
    
    # Create a synthetic binary PLE table on disk
    table_bytes = bytearray(row_layout.bytes_for(total_physical_rows))
    
    # Write a test row (row 5)
    row_5_offset = row_layout.row_offset(5)
    test_packed = bytes([0x88] * 80)  # 4-bit unpacked -> 0.0
    test_scale = torch.tensor([1.5], dtype=torch.float16).numpy().tobytes()
    test_row_bytes = test_packed + test_scale
    
    table_bytes[row_5_offset : row_5_offset + len(test_row_bytes)] = test_row_bytes
    
    with tempfile.TemporaryDirectory() as tmpdir:
        table_path = Path(tmpdir) / "table.bin"
        table_path.write_bytes(table_bytes)
        
        with PleRowStore(table_path, dummy_ple_index, cache_capacity_rows=10) as store:
            row_bytes_read = store.read_row_raw(5)
            assert row_bytes_read == test_row_bytes
            
            # Dequantize
            tensor = store.decode_row(row_bytes_read)
            assert tensor.shape == torch.Size([160])
            assert tensor.dtype == torch.float16
            assert torch.allclose(tensor, torch.zeros(160, dtype=torch.float16))
            
            # Cache test
            t1 = store.fetch_row(5)
            assert 5 in store.cache
            t2 = store.fetch_row(5)  # Cache hit
            assert torch.equal(t1, t2)
