"""Tests for PLE runtime hasher and SSD row store (R5)."""

import tempfile
from pathlib import Path
import pytest
import torch

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.package.format import (
    PleIndex,
    PleRowLayout,
    Section,
    section_spans,
)
from pockettitan.quantizers import get_quantizer
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


def _ple_index(row_layout: PleRowLayout) -> PleIndex:
    num_heads = 16
    return PleIndex(
        ngram_size=3,
        num_heads=num_heads,
        heads_per_ngram=8,
        head_offsets=[i * 100000 for i in range(num_heads)],
        head_vocab_sizes=[20000003 + i * 10 for i in range(num_heads)],
        layer_multipliers=[23703573157769, 20109073645365, 8052911324071],
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


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("symmetric", [True, False])
def test_ple_row_store_decodes_what_the_quantizer_wrote(bits, symmetric):
    """The store must decode rows the *packer* produced, not rows shaped to suit it.

    The earlier version of this test wrote ``b"" * 80`` — a byte pattern
    chosen so that the decoder's hardcoded "nibble minus 8" arithmetic returned
    zero. That is self-consistent, not a round-trip: it passed while the decoder
    disagreed with every layout the planner actually emits. Here the row is
    produced by the real quantizer at the real `PT-Q4E` setting (3 bits), so a
    decoder that assumes 4-bit nibbles or the wrong zero offset fails.
    """
    row_width = 160
    row_layout = PleRowLayout.build(
        row_width=row_width, bits=float(bits), group_size=row_width, symmetric=symmetric
    )
    index = _ple_index(row_layout)

    torch.manual_seed(bits)
    source = torch.randn(1, row_width, dtype=torch.float32)
    config = QuantConfig(
        method=QuantMethod.RTN, bits=bits, group_size=row_width,
        symmetric=symmetric, device="cpu",
    )
    quantizer = get_quantizer(config)
    result = quantizer.quantize(source)

    payload = bytearray(row_layout.payload_bytes)
    sections = {
        Section.PACKED: result.packed_weights.numpy().tobytes(),
        Section.SCALES: result.scales.numpy().tobytes(),
        Section.ZEROS: result.zeros.numpy().tobytes() if result.zeros is not None else b"",
    }
    for span in section_spans([1, row_width], float(bits), row_width, symmetric):
        blob = sections[span.section]
        assert len(blob) == span.length, f"{span.section} is {len(blob)}B, planned {span.length}B"
        payload[span.offset : span.offset + span.length] = blob

    table = bytearray(row_layout.bytes_for(100))
    offset = row_layout.row_offset(5)
    table[offset : offset + len(payload)] = payload

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "table.bin"
        path.write_bytes(table)
        with PleRowStore(path, index, cache_capacity_rows=10) as store:
            assert store.read_row_raw(5) == bytes(payload)
            decoded = store.decode_row(store.read_row_raw(5))

    expected = quantizer.dequantize(result).flatten().float()
    assert decoded.shape == (row_width,)
    assert torch.allclose(decoded.float(), expected, atol=1e-2), (
        f"decoded row does not match the quantizer's own reconstruction "
        f"(max diff {(decoded.float() - expected).abs().max():.4f})"
    )
    # And it must actually resemble the source, not merely agree with itself.
    assert torch.corrcoef(torch.stack([decoded.float(), source.flatten()]))[0, 1] > 0.9


def test_ple_rows_never_straddle_a_page(dummy_ple_index):
    row = dummy_ple_index.row
    for row_id in range(0, 500):
        start = row.row_offset(row_id)
        assert start // row.page_bytes == (start + row.payload_bytes - 1) // row.page_bytes
