"""The single decode path from packaged bytes back to weights.

Every consumer of a ``.ptitan`` region — the PLE row store, the expert manager,
the dense blob reader — needs to turn ``(payload, spans, geometry)`` into a
tensor. Each one previously hand-rolled that, and each hand-rolled version drifted
from what :class:`PackageWriter` actually emits:

* the PLE store assumed 4-bit nibbles and a ``-8`` offset. ``PT-Q4E`` writes the
  table at 3 bits, where codes live in bits 0-2 and 3-5 and the symmetric offset
  is ``max_int // 2 == 3``. Decoded rows correlated 0.247 with the source.
* the expert manager ignored the ``ZEROS`` section entirely, although
  ``PrecisionEntry.symmetric`` defaults to ``False``, and sliced rows at the
  unpadded width so every row after the first was misaligned.

Routing everything through the quantizer that produced the bytes makes those
drifts impossible: the reconstruction is the writer's own inverse by
construction, not a second implementation of it.
"""

from typing import Optional, Sequence, Tuple

import torch

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.package.format import (
    Section,
    SectionSpan,
    matrix_dims,
    packed_bytes,
    section_spans,
)
from pockettitan.quantizers import get_quantizer
from pockettitan.quantizers.base import QuantizedResult


class DecodeError(ValueError):
    """The payload does not match the geometry it was supposed to be written with."""


def decode_record(
    payload: bytes,
    shape: Sequence[int],
    bits: float,
    group_size: int,
    symmetric: bool,
    spans: Optional[Sequence[SectionSpan]] = None,
    method: QuantMethod = QuantMethod.RTN,
    base_offset: int = 0,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct one quantized tensor from its packed bytes.

    ``spans`` defaults to the canonical layout for this geometry, which is what
    the planner used to size the record. Pass the manifest's spans when reading
    a record whose sections were placed by the planner.
    """
    shape = tuple(int(d) for d in shape)

    if bits >= 16:
        want = _num_elements(shape) * 2
        raw = payload[base_offset : base_offset + want]
        if len(raw) != want:
            raise DecodeError(f"fp16 payload is {len(raw)} bytes, expected {want}")
        return torch.frombuffer(bytearray(raw), dtype=torch.float16).view(*shape).to(dtype)

    if spans is None:
        spans = section_spans(list(shape), bits, group_size, symmetric)
    by_section = {span.section: span for span in spans}

    if Section.PACKED not in by_section:
        raise DecodeError("record has no PACKED section")

    def _read(section: Section, torch_dtype: torch.dtype) -> Optional[torch.Tensor]:
        span = by_section.get(section)
        if span is None:
            return None
        start = base_offset + span.offset
        raw = payload[start : start + span.length]
        if len(raw) != span.length:
            raise DecodeError(
                f"{section.value} section wants bytes [{start}, {start + span.length}) "
                f"but the payload holds {len(payload)}"
            )
        return torch.frombuffer(bytearray(raw), dtype=torch_dtype)

    config = QuantConfig(
        method=method,
        bits=int(bits),
        group_size=group_size,
        symmetric=symmetric,
        device="cpu",
    )
    result = QuantizedResult(
        packed_weights=_read(Section.PACKED, torch.uint8),
        scales=_read(Section.SCALES, torch.float16),
        zeros=_read(Section.ZEROS, torch.float16),
        codebook=None,
        quant_config=config,
        original_shape=shape,
        original_dtype=dtype,
        bit_width=float(bits),
        device="cpu",
    )
    if not symmetric and result.zeros is None:
        raise DecodeError(
            "asymmetric record is missing its ZEROS section; decoding it as symmetric "
            "would apply a constant offset to every value"
        )
    return get_quantizer(config).dequantize(result).view(*shape).to(dtype)


def _num_elements(shape: Tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def row_slice_is_byte_aligned(in_features: int, bits: float) -> bool:
    """Whether one matrix row starts on a byte boundary in the PACKED section.

    ``_pack_tensor`` packs the flattened matrix, so a row begins mid-byte unless
    its width is a whole number of bytes at the storage width. 5120 and 17408 are
    both fine at every width the planner emits; odd widths are not.
    """
    if bits >= 8:
        return True
    values_per_byte = 8 // int(bits)
    return in_features % values_per_byte == 0


def decode_rows(
    payload: bytes,
    shape: Sequence[int],
    bits: float,
    group_size: int,
    symmetric: bool,
    row_start: int,
    row_stop: int,
    spans: Optional[Sequence[SectionSpan]] = None,
    method: QuantMethod = QuantMethod.RTN,
    base_offset: int = 0,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reconstruct rows ``[row_start, row_stop)`` without touching the rest.

    ``embed_tokens`` and ``lm_head`` are 248,320 x 5,120 on the 27B — 2.5 GB each
    once materialized. Both are row-addressed in use (an embedding lookup reads a
    handful of rows; a logit matmul can be chunked), so decoding the whole matrix
    to use part of it is what makes the difference between fitting in 12 GB and
    not.
    """
    shape = tuple(int(d) for d in shape)
    rows, in_features = matrix_dims(list(shape))
    if not 0 <= row_start <= row_stop <= rows:
        raise DecodeError(f"row range [{row_start}, {row_stop}) is outside {rows} rows")
    if row_start == row_stop:
        return torch.empty((0, in_features), dtype=dtype)

    if bits >= 16:
        stride = in_features * 2
        start = base_offset + row_start * stride
        raw = payload[start : start + (row_stop - row_start) * stride]
        return (
            torch.frombuffer(bytearray(raw), dtype=torch.float16)
            .view(row_stop - row_start, in_features)
            .to(dtype)
        )

    if not row_slice_is_byte_aligned(in_features, bits):
        whole = decode_record(
            payload, shape, bits, group_size, symmetric, spans, method, base_offset, dtype
        )
        return whole.reshape(rows, in_features)[row_start:row_stop]

    if spans is None:
        spans = section_spans(list(shape), bits, group_size, symmetric)
    by_section = {span.section: span for span in spans}

    effective_group = group_size if group_size > 0 else in_features
    if in_features % effective_group:
        raise DecodeError(
            f"group_size={group_size} does not divide in_features={in_features}; "
            "the planner resolves group sizes to divisors"
        )
    groups_per_row = in_features // effective_group
    count = row_stop - row_start

    packed_stride = packed_bytes(in_features, bits)
    packed_span = by_section[Section.PACKED]
    packed_start = base_offset + packed_span.offset + row_start * packed_stride
    packed = torch.frombuffer(
        bytearray(payload[packed_start : packed_start + count * packed_stride]), dtype=torch.uint8
    )
    if packed.numel() != count * packed_stride:
        raise DecodeError("PACKED section is shorter than the requested rows")

    def _meta(section: Section) -> Optional[torch.Tensor]:
        span = by_section.get(section)
        if span is None:
            return None
        stride = groups_per_row * 2
        start = base_offset + span.offset + row_start * stride
        raw = payload[start : start + count * stride]
        if len(raw) != count * stride:
            raise DecodeError(f"{section.value} section is shorter than the requested rows")
        return torch.frombuffer(bytearray(raw), dtype=torch.float16)

    config = QuantConfig(
        method=method,
        bits=int(bits),
        group_size=effective_group,
        symmetric=symmetric,
        device="cpu",
    )
    result = QuantizedResult(
        packed_weights=packed,
        scales=_meta(Section.SCALES),
        zeros=_meta(Section.ZEROS),
        codebook=None,
        quant_config=config,
        original_shape=(count, in_features),
        original_dtype=dtype,
        bit_width=float(bits),
        device="cpu",
    )
    return get_quantizer(config).dequantize(result).view(count, in_features).to(dtype)
