# R1a — Package Format, Slicing & Build Planner

```text
PHASE:        R1a (planning half of R1 — Packager v1)
GATE METRIC:  expert record payload x records = audit estimate
              2,611,200 x 24,576 = 64,172,851,200  (exact match, two independent derivations)
              live slice decode = 6/6 experts well-formed bf16
DECISION:     PROCEED to R1b (writer)
EVIDENCE:     reports/r1-plan-pt-q4e.json · tests/test_package.py (48 offline + 1 network)
SURPRISES:    1 (test fixture stored data-relative offsets — regenerated)
NEXT:         R1b — streaming writer, then T1.7 defect fixes
```

## What shipped

| Module | Responsibility |
| :--- | :--- |
| `pockettitan/package/format.py` | On-disk layout: record/row geometry, manifest schema, PLE index with the row hash |
| `pockettitan/package/slicing.py` | `(layer, expert) -> byte ranges` for fused **and** per-expert checkpoints (T1.2) |
| `pockettitan/package/plan.py` | Byte-exact `BuildPlan` from headers alone (T1.1, T1.3, T1.4, T1.5) |
| `pockettitan/package/report.py` | Rich rendering |
| `pockettitan/cli.py::plan` | `pockettitan plan <MODEL> [--precision] [--features] [-o plan.json]` |

```bash
pockettitan plan Qwen/Qwen3.8-Flash-Next --precision pt-q4e -o reports/r1-plan-pt-q4e.json
```

## The planned PT-Q4E package

| Region | Contents | Items | Bytes |
| :--- | :--- | ---: | ---: |
| `dense/` | VRAM-resident core | 1,070 | 2.44 GiB |
| `experts/` | 512 experts × 48 layers, 2.49 MiB/record | 24,576 | 59.81 GiB |
| `ple/` | 320,001,536 rows × 64 B stride (64/page) | 128 | 19.07 GiB |
| **TOTAL** | | **25,774** | **81.33 GiB** |

Dropped: 3,056,081,904 params (vision + MTP). Source bytes to read: 329.58 GiB — we do not
download what we discard.

### Expert record

```text
gate_up_proj  1280x2560  @4b   offset        0   packed 1,638,400 + scales 51,200 + zeros 51,200
down_proj     2560x640   @4b   offset 1,740,800  packed   819,200 + scales 25,600 + zeros 25,600
RECORD        4,915,200 params                   payload 2,611,200 B   stride 2,613,248 B (4096-aligned)
```

Layer-major ordering keeps a layer's 512 experts adjacent, so a layer's working set is a
contiguous 1.28 GiB span rather than 512 scattered reads across the bank.

## Verification

**Cross-derivation agreement.** The packager's explicit per-record accounting
(`packed + scales + zeros`, summed over 24,576 records) equals the R0 audit's amortized
`params × effective_bits` estimate *exactly*: 64,172,851,200 bytes. Two independent
derivations of the same quantity, asserted in `test_expert_payload_matches_audit_budget`.

**Live checkpoint decode.** Experts 0, 300 and 511 of layer 0 were fetched from the CDN using
computed byte offsets and decoded as bf16:

| Slice | Shape | Bytes | Finite | std | zeros |
| :--- | :--- | ---: | :--- | ---: | ---: |
| L0 E0 gate_up | 1280×2560 | 6,553,600 | yes | 0.01345 | 0.000 |
| L0 E0 down | 2560×640 | 3,276,800 | yes | 0.01257 | 0.000 |
| L0 E300 gate_up | 1280×2560 | 6,553,600 | yes | 0.01405 | 0.000 |
| L0 E511 down | 2560×640 | 3,276,800 | yes | 0.01147 | 0.000 |

bf16 is unforgiving about alignment: an offset wrong by one byte reinterprets every subsequent
pair and produces denormals and infinities. Well-conditioned weights are therefore proof of
correct addressing. Codified as `test_expert_slices_decode_from_live_checkpoint`, marked
`network` and deselected by default.

**Alignment cost.** Page alignment adds 0.078% (2,048 B per 2,611,200 B record) — asserted
below 0.1% so the invariant cannot silently regress.

## Design decisions worth knowing

**Planning is separate from writing.** `plan_package()` reads zero weights. That is what makes
the build resumable (every output byte has a known address before the first read), reviewable
(the plan is a diffable JSON artifact), and verifiable (totals check against R0 before anything
is downloaded).

**PLE row stride is a power of two.** 62 B of payload rounds to a 64 B stride, so 4096 % 64 == 0
and a row can never straddle a page. Costs 3.2% capacity (18.48 → 19.07 GiB) and halves the
worst-case page count of every lookup. `test_ple_rows_never_straddle_a_page` asserts this
directly rather than trusting the arithmetic.

**Slicing refuses to guess.** A bank whose leading axis disagrees with `num_experts`, a layer
mixing fused and per-expert tensors, or experts with inconsistent geometry all raise
`SliceError`. Each of those means the caller is about to read the wrong bytes; silently
proceeding would produce a package that looks valid and is numerically garbage.

## Surprises

**The R0 golden fixture stored data-section-relative offsets while `scan_checkpoint` produces
absolute ones.** Every R0 test passed because they only did relative arithmetic, so the flaw was
invisible until R1 fetched real bytes and had to add the shard header length by hand. A fixture
that misrepresents what it stands in for will eventually cause a real bug, so it was regenerated
with absolute offsets (16 KB → 22 KB, now also carrying per-shard `header_bytes`). The network
test then passed unchanged, confirming the fixture is a faithful reproduction of a live scan.

## Status of R1

Done: T1.1 – T1.5 in planning and addressing form, plus the CLI.

Remaining for R1b:

- **T1.6 Writer** — stream, quantize, emit `dense/` + `experts/bank.bin` + `ple/table.bin` +
  `manifest.json`, resumable, peak VRAM < 3.5 GiB.
- **T1.7 Defects (§4.1)** — the PLE shard OOM is the blocker: `(2500012, 160)` tensors are not
  being tiled, so the writer would OOM on the same tensors the old pipeline did.
- **T1.8** — extend `pockettitan validate` to the `.ptitan` format.

The R1 gate ("package built end-to-end, peak VRAM < 3.5 GiB, resumable, validate passes") is
**not** met yet; it belongs to R1b.
