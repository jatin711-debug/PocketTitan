# R1b (T1.6) — Streaming Package Writer

```text
PHASE:        R1b task T1.6 - build the .ptitan package
GATE METRIC:  round-trip: every region read back from planned offsets and dequantized
              region file sizes == planned sizes, exactly, for all three regions
              resumable: partial journal completes without redoing finished work
DECISION:     PROCEED to T1.8 (validate) / T1.9 (dense 3-bit packing)
EVIDENCE:     tests/test_writer.py (18) + tests/test_package.py (51) · reports/r1-plan-pt-q4e.json
SURPRISES:    2 (sub-byte packing density; group padding in byte accounting)
NEXT:         T1.8 validate for .ptitan, then R2 simulator
```

## What shipped

| Module | Responsibility |
| :--- | :--- |
| `package/writer.py` | Executes a `BuildPlan`: read slice → quantize under VRAM cap → write at planned offset |
| `package/format.py` | `storage_bits`, `section_spans` with padding, page-packed `PleRowLayout` |
| `package/plan.py` | Dense byte offsets, packing/padding warnings |
| `cli.py::package` | `pockettitan package <MODEL> <OUT.ptitan>` with progress and resume |

```bash
pockettitan package Qwen/Qwen3.8-Flash-Next ./qwen38.ptitan --precision pt-q4e --max-vram 3584MB
```

Three properties make the build safe on a 12 GB machine producing an 87 GiB package:

- **Every output address is known before the build starts.** Regions are preallocated to their
  planned size and records are written by seek-and-write. Nothing is buffered waiting for a
  neighbour, so peak memory is one work item.
- **Resumable at item granularity.** The journal is keyed on a *layout fingerprint*; resuming
  into a package built from a different plan is refused rather than silently corrupting it.
- **Bounded residency.** All quantization routes through `MatrixTiler`, which enforces the VRAM
  ceiling fixed in T1.7.

## Verification

The load-bearing tests are round-trips: weights go in at planned offsets, then come back out
**using only the manifest** and get dequantized. If an offset or a section length is wrong the
reconstruction is garbage rather than merely imprecise, so an approximate comparison is a sharp
test.

| Test | Proves |
| :--- | :--- |
| `test_expert_roundtrip_from_bank` | `(layer, expert)` → byte range → dequantized weights match source within 4-bit error, for experts at both ends and the middle |
| `test_expert_records_are_distinct` | All 8 records differ — catches every record landing on one offset |
| `test_dense_roundtrip_from_blob` | Dense entries addressable by `byte_offset` |
| `test_fp16_components_are_stored_verbatim` | Routers survive **bit-exact** — a wrong top-k is unrecoverable |
| `test_ple_table_roundtrip` | Every row independently decodable from its own page-packed offset |
| `test_resume_completes_a_partial_build` | Truncated journal → only the dropped items are redone |
| `test_resume_rejects_a_different_layout` | Fingerprint mismatch raises instead of corrupting |
| `test_section_size_mismatch_is_fatal` | A short write raises rather than shifting every later record |

185 offline tests pass, plus 2 `gpu`-marked and 1 `network`-marked.

## Surprises

**1. Sub-byte packing is only dense for 1, 2, 4, and 8 bits.** The packer stores `8 // bits`
values per byte, so 3-bit occupies **4** bits on disk and 6-bit occupies **8**. `PT-Q4E`
requested 3-bit for GDN and the PLE table and 6-bit for `lm_head`, so the previously reported
81.33 GiB was fiction. Measured reality:

| | planned before | actual |
| :--- | ---: | ---: |
| GDN (3-bit) | 0.79 GiB | 1.055 GiB |
| `lm_head` (6-bit) | 0.46 GiB | 0.611 GiB |
| PLE table (3-bit) | 18.48 GiB | 24.91 GiB |
| **Package total** | **81.33 GiB** | **87.61 GiB** |
| **VRAM-resident dense** | **2.12 GiB** | **2.57 GiB** |

The format now models this (`storage_bits`) and the planner warns. Plan.md's `PT-Q4E` table was
corrected in the same pass, per the §9 reporting rule.

The VRAM figure matters more than the disk figure: 2.57 GiB of a ~3.5 GiB usable budget leaves
much less room for the VRAM hot-expert tier than the research report assumed — which is exactly
the risk it flagged under "4 GB VRAM is genuinely tight". **T1.9 (dense 3-bit packing) would
recover ~6.5 GiB on disk and 0.27 GiB of VRAM**, and is now the highest-value packer follow-up.

**2. Group padding again, this time in byte accounting.** `group_size=128` on a 64-wide row
pads to 128 and *doubles* the stored bytes. This is the same defect class as the T1.7 tiler OOM,
in a different subsystem: the tiler modelled padding for VRAM, but `section_spans` sized records
from the unpadded element count. A short section would have shifted every subsequent record.
Fixed by giving `section_spans` the shape rather than an element count, and the planner now
warns:

```text
! experts_routed.down_proj: in_features=64 is not a multiple of group_size=128,
  so it is stored 2.00x larger. Set group_size to a divisor of 64.
```

Qwen3.8's real shapes (2560, 640, 160) are all clean under their assigned group sizes, so this
does not affect the target model — but it would silently double storage on any checkpoint whose
widths are not multiples of 128.

## Design decision: page-packed PLE rows

Rounding the row stride up to a power of two — the original design — costs 36% for an 82-byte
row (128-byte stride). Page packing instead fits whole rows into each 4 KiB page and leaves the
remainder unused: 49 rows/page, **1.9% waste**, and a row still never straddles a page. On the
real table that is 24.91 GiB instead of 38 GiB.

## A test gap this phase exposed

The live `plan` command crashed in `render_regions` on a stale `row.stride` reference, *after*
the full suite passed. Cause: no test rendered a plan that **had** a PLE region — the synthetic
MoE fixture has no n-gram table, and the Qwen fixture tests never called the renderer. Fixed,
and covered by `test_render_plan_covers_every_region`, which asserts all three region branches
are exercised. Worth remembering: a renderer is only tested by the shapes of data you actually
render.

## Remaining for R1

- **T1.8** — extend `pockettitan validate` to the `.ptitan` format.
- **T1.9** — dense 3-bit packing (~6.5 GiB disk, 0.27 GiB VRAM).
- **T1.10** — copy `tokenizer/` into the package, then run a full build against the live
  checkpoint. The R1 gate ("built end-to-end from the remote checkpoint, peak VRAM < 3.5 GiB")
  needs that run; everything it depends on is now in place and verified on synthetic models.
