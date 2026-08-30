# R1b (T1.7) — Group-Padding VRAM Accounting

```text
PHASE:        R1b task T1.7 - unblock the writer
GATE METRIC:  PLE shard (2500012,160) tiles instead of OOM
              worst-case tile peak = 1992 MiB   (budget: 2688 MiB usable)
DECISION:     PROCEED to T1.6 (writer)
EVIDENCE:     tests/test_tiler.py (29 offline + 1 gpu-marked)
SURPRISES:    2 (padding never modelled; all 7 multipliers under-declared)
NEXT:         T1.6 - streaming package writer
```

## Root cause

The pipeline recorded `CUDA out of memory. Tried to allocate 2.38 GiB` on every
`ngram_embedding.shard_*` tensor. Those are `(2500012, 160)` BF16 — 800 MB of source
data on a card with 2.62 GiB usable. The tiler had decided no tiling was needed.

Two independent defects compounded:

**1. Group-size padding was never modelled.** Group-wise quantizers pad the input dimension
up to a multiple of `group_size` *before* grouping. With `group_size=128` a 160-wide row is
materialized as 256 wide — a **1.60× blowup of every fp32 intermediate**:

```text
400,001,920 elements  ->  640,003,072 padded  ->  fp32 copy = 2.38 GiB
```

That is precisely the allocation the driver refused. The estimator, unaware of padding,
predicted 1.74 GiB, passed it as fitting, and handed the whole tensor to the GPU.

**2. Every quantizer under-declared its workspace multiplier.** Measured peak-to-source ratios
on an RTX 3050, on group-aligned matrices:

| Quantizer | Declared | Measured | Now |
| :--- | ---: | ---: | ---: |
| ternary | 1.2 | 6.51 | 7.0 |
| rtn | 2.0 | 6.05 | 6.5 |
| intx | 1.5 | 6.05 | 6.5 |
| hqq | 5.0 | 12.09 | 13.0 |
| gptq | 3.0 | 12.39 | 13.5 |
| awq | 2.5 | 14.09 | 15.0 |
| autoround | 3.5 | 34.06 | 36.0 |

Every one was optimistic by 3–6×. `ternary` — the backend used for the failing run — was the
worst relative offender at 5.4×.

## The insight that shaped the fix

Measuring both a padded and an aligned matrix for each quantizer gave ratios differing by
**exactly 1.60** in every case (ternary 10.43 vs 6.51; rtn 9.70 vs 6.05; hqq 19.35 vs 12.09).
That is the padding factor, not quantizer-specific behaviour.

So padding and workspace are **separable**: model padding explicitly and the workspace
multiplier becomes shape-independent. That is why the fix is a multiplicative
`group_padding_factor()` term rather than per-quantizer fudge factors — the multiplier now
means one thing ("peak/source on an aligned matrix") and can be measured once.

## Changes

- `scheduler/budget.py`
  - `group_padding_factor(in_features, group_size)` — new.
  - `source_dtype_bytes(dtype)` — new; replaces substring matching that mis-sized `I64`
    (matched neither `"32"` nor `"8"`, so it returned 2 bytes for an 8-byte dtype).
  - `estimate_tensor_vram_requirement(..., group_size=0)` — applies the padding factor.
  - `compute_work_unit_bounds` — passes `group_size` through and uses padded per-row cost.
- All seven quantizers: `workspace_multiplier` set to measured values, with the measurement
  recorded in a comment beside each.

## Verification

```text
before:  needs_tiling=False  ->  single 400M-element dispatch  ->  OOM at 2.38 GiB
after :  group_size=128  needs_tiling=True  tiles=5  est 2150 MiB/tile
         group_size=160  needs_tiling=True  tiles=3  est 2150 MiB/tile

real execution, RTX 3050, 570,944 x 160 tile:
         peak VRAM = 1992 MiB   budget = 2688 MiB   OK
```

The estimate (2150 MiB) exceeds the measured peak (1992 MiB) by 8% — conservative in the
correct direction.

Tests added to `tests/test_tiler.py`:

- `test_ple_shard_requires_tiling` — the exact regression, both group sizes.
- `test_declared_workspace_multiplier_is_not_optimistic` — parametrized over all seven
  backends, asserting declared > measured. This is the guard that stops the class of bug
  recurring, not just this instance.
- `test_group_padding_factor`, `test_source_dtype_bytes`, `test_estimator_accounts_for_group_padding`
- `test_tighter_group_size_needs_fewer_tiles`
- `test_ple_shard_tile_fits_real_vram` — `gpu`-marked, executes on the real device.

`pytest` markers `network` and `gpu` are registered and deselected by default.

## Consequence for the packager

`PT-Q4E` already assigns `group_size=160` to `PLE_TABLE` — one group per row, so the padding
factor is 1.0 and the blowup never occurs in a planned package. The generic tiler needed the
fix anyway because the writer routes every tensor through it, and any future component whose
width is not a multiple of 128 would have hit the same wall.

Worth noting for R1's precision map: `group_size=128` on a 160-wide row is not merely a memory
problem — 37.5% of each padded group is zero-fill, which distorts the group scale. Matching
group size to row width is both cheaper and more accurate.
