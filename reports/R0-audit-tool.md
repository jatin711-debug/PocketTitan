# R0 — Audit Tool

```text
PHASE:        R0 Audit Tool
GATE METRIC:  total_params = 179,999,981,459   (threshold: exact match)
              lm_core      = 125,743,653,795   (threshold: == llama.cpp's 125.74 B)
              cold scan    = 18.3 s scan / 28.7 s wall   (threshold: < 60 s)
DECISION:     PROCEED
EVIDENCE:     reports/r0-qwen38-pt-q4e.json · tests/test_audit.py (49 passed)
              tests/data/qwen38_flash_next_headers.json.gz (golden fixture, 16 KB)
SURPRISES:    1 (taxonomy boundary — Plan.md §2.1 updated in this commit)
NEXT:         R1 — Packager v1
```

## What shipped

| Module | Responsibility |
| :--- | :--- |
| `pockettitan/audit/headers.py` | Strict parallel Safetensors header scan with retry/backoff, plus three-way verification against the published index |
| `pockettitan/audit/classify.py` | Ordered first-match taxonomy resolving component + capability + tier + activation mode |
| `pockettitan/audit/budget.py` | Activation, storage, state, and roofline budgets; `PT-Q4E` / `PT-Q2E` precision maps |
| `pockettitan/audit/report.py` | Rich rendering with encoding-adaptive glyphs |
| `pockettitan/cli.py::audit` | `pockettitan audit <MODEL> [--precision] [--features] [-o report.json]` |

```bash
pockettitan audit Qwen/Qwen3.8-Flash-Next --precision pt-q4e --features text -o reports/r0.json
```

## Gate verification

All figures reproduced from the live checkpoint (`Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d`), and pinned
offline by the golden fixture so CI does not need network access.

| Quantity | Measured | Cross-check |
| :--- | ---: | :--- |
| Total parameters | 179,999,981,459 | index `total_size` / 2 B |
| Summed tensor bytes | 359,999,963,128 | equals published `total_size` exactly |
| LM core (text, excl. n-gram table) | 125,743,653,795 | llama.cpp reports `125.74 B` |
| Activated params / token | 6,671,300,515 | 3.71% of model |
| Routed-expert params / token | 2,359,296,000 | 1.31% of model |
| Expert record @ 4.25 bits | 2.49 MiB | 480 reads/token |
| Dense core resident in VRAM | 2.12 GiB | fits 4 GB with state + staging |
| `PT-Q4E` package total | 80.68 GiB | 3.92 bits/param average |
| Cache capacity @ 7 GiB RAM | 2,878 / 24,576 slots (11.7%) | **the open risk — Q1** |

Tensors classified: 1,658 / 1,658. No unclassified tensors, no index discrepancies.

## Design decisions worth knowing

**Budgets are derived from measured tensor shapes, not architecture constants.** Activated
parameters per token are computed by applying an activation factor to each component
(`DENSE` → 1.0, `ROUTED` → `top_k/num_experts`, `ROW_LOOKUP` → rows touched), so the figure
cannot drift from the checkpoint the way a hand-written layer formula would. Only routing
topology is read from config, because it is not recoverable from shapes.

**Strict by default.** `scan_checkpoint` raises `ShardScanError` if any shard fails. A partial
scan that looks successful would silently understate every budget; `--no-strict` exists for
exploration only and records each failure as a discrepancy.

**`enabled_params` vs `lm_core_params`.** These differ by 51.2B and conflating them is a real
hazard. The n-gram table is *text* capability — it is part of the language model — but it is
stored as a separate artifact, which is why conversion tools report 125.74 B. Both are exposed.

## Surprises

**Taxonomy boundary: `shared_expert_gate` reclassified from `SHARED_EXPERT` to `ROUTER`.**
The 48 `mlp.shared_expert_gate.weight` tensors (122,880 params total) are gates, and gates must
stay fp16. Classifying them with the shared expert would have silently quantized them to 4-bit
under `PT-Q4E`. Component totals shift by 122,880 params; the model total is unchanged.
Plan.md §2.1 updated in the same commit, per the §9 reporting rule.

## Defects found and fixed while building

- **`UnicodeEncodeError` on Windows.** Rendering crashed mid-report when stdout was a cp1252
  stream (i.e. any redirected output on a default Windows console). Fixed in two layers: the CLI
  reconfigures stdio to UTF-8, and the renderer resolves its glyph set against the destination
  encoding so it degrades instead of raising. Regression test:
  `test_render_survives_legacy_codepage`.
- **Truncated parameter counts.** Redirected output fell back to 80 columns, rendering
  `120,795,9…`. An audit that elides digits is not an audit; non-interactive output is now 160
  columns.
- **Arbitrary precision in the capability table.** Packed sizes for dropped components used
  `entries[0].effective_bits`. `AuditReport` now carries its `PrecisionMap` and looks up the real
  per-component assignment.

## Carried into R1

Unchanged from Plan.md §4.1 — R0 measured these but did not fix them:

- PLE shard OOM in the quantization pipeline (`(2500012, 160)` tensors are not being tiled).
- Per-expert addressing is tensor-granular; `ExpertGeometry.tensors_per_expert == 2` confirms an
  expert is a slice of two fused tensors, so R1's repacker must produce one contiguous record.
- `inspect`'s parameter math still divides `total_size` by dtype width. The checkpoint is mixed
  dtype (1,655 BF16 + 3 I64), so that estimate is structurally wrong; `audit` supersedes it.
