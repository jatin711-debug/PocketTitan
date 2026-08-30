# R1d — Pre-Rebuild Defect Sweep

```text
PHASE:        R1d - fix everything broken in the build and decode paths
GATE METRIC:  every packaged region decodes back to its source through ONE decoder
              ruff clean; no record padded; kernels match the packer's arithmetic
DECISION:     the build path is sound; rebuild from a fresh directory
EVIDENCE:     287 offline tests (224 passing / 6 failing on entry) - ruff: All checks passed
SURPRISES:    the three worst bugs were all in decoders, and all three had tests
NEXT:         rebuild, then a reference forward pass (FORK GATE)
```

## The pattern behind almost every defect

Seven of the nine bugs below are the same mistake: **a second implementation of
the packer's inverse**. Each consumer — the PLE row store, the expert manager,
the LUT kernel, the fused CUDA kernel, `demo_chat.py` — reimplemented "unpack
codes, subtract an offset, multiply by scale". Each drifted, and drifted
*silently*, because each was tested against a fixture built to match its own
assumption rather than against bytes the writer produced.

The structural fix is [`pockettitan/package/decode.py`](../pockettitan/package/decode.py):
one `decode_record(payload, shape, bits, group_size, symmetric, spans)` that
routes through the quantizer that produced the bytes. It is the writer's own
inverse by construction, so it cannot disagree with the writer. Everything now
calls it.

## Fixed

| # | Defect | Consequence |
| :-- | :--- | :--- |
| 1 | `google_crc32c.Checksum` has no `.value`, and its constructor takes leading *data*, not a CRC seed | 6 failing tests; a package written with the fast backend installed would fail validation without it |
| 2 | `dequantize` derived dims as `(shape[0], shape[1])` while `quantize` flattens with `view(-1, shape[-1])` — all 7 quantizers | Every non-2-D tensor unreadable: 224 tensors in the 27B package |
| 3 | The affine zero-point was clamped into `[0, max_int]` (RTN, HQQ ×2) | A group that does not straddle zero loses all resolution. `linear_attn.norm.weight`: 128 values → **one constant** |
| 4 | `linear_attn.` / `self_attn.` prefix rules outranked the norm rule | `A_log` → 6 distinct values, `dt_bias` → 8, attention norms → 1. Same bug shape as `shared_expert_gate` in R0 |
| 5 | `PleRowStore.decode_row` assumed 4-bit nibbles and a `-8` offset | `PT-Q4E` writes the table at **3** bits. Decoded rows correlated **0.247** with the source — 24.9 GiB of noise |
| 6 | `decode_expert_payload` ignored the `ZEROS` section and sliced at the unpadded width | `symmetric` defaults to `False`, so every packaged expert has a zero-point. On real bytes it **raised** |
| 7 | `cpu_lut` / `cuda_fused` used `code - 8` (4-bit) and `code - 2` (2-bit), with no zero-point | RTN centres on `max_int // 2` = 7 and 1. Every weight biased one code; asymmetric records inexpressible |
| 8 | Group sizes that do not divide `in_features` padded the record | `conv1d.weight` (4 inputs, `group_size=128`): 3.9 MB of fp16 became **33.4 MB**, with scales set by 124 padding zeros |
| 9 | `json` was never imported in `cli.py` | `pockettitan profile prompts --output` raised `NameError` |

Plus: two blanket `except TypeError` retries in the writer that existed only to
tolerate test stubs with the wrong signature — they would have swallowed a real
`TypeError` from inside an encoder and silently redone the work. Removed; the
stubs were corrected instead.

## Group padding is now impossible, not merely warned about

`resolve_group_size(in_features, group_size)` rounds the requested size down to a
divisor. Padding cost storage *and* accuracy — the padding zeros joined the group
and stretched its scale over a range the real weights never occupy — so rounding
down is strictly better on both counts. The planner now **raises** if any record
would be padded, and reports where a size was reduced:

```text
! hyperconn: group_size 128 does not divide in_features=320,
  so it was reduced to 80 rather than padding the record.
```

## Verification

Every fix is pinned by a test that fails against the old code:

- `test_seeded_crc32c_continues_a_running_checksum` — chained == one-shot == reference
- `test_dequantize_restores_non_2d_shapes` — 1-D and 3-D through every uncalibrated backend
- `test_narrow_band_group_away_from_zero_keeps_resolution` — must not collapse to a constant
- `test_norms_and_state_params_outrank_the_attention_family_rules`
- `test_ple_row_store_decodes_what_the_quantizer_wrote` — the row is produced by the real packer at the real 3-bit setting
- `test_expert_manager_decodes_a_package_the_writer_actually_wrote` — experts read from a real `bank.bin`
- `test_kernels_agree_with_the_packers_own_dequantization`
- `test_no_planned_record_is_ever_padded`

Measured, old vs new, on bytes the writer produced:

| Path | Before | After |
| :--- | :--- | :--- |
| PLE row, 3-bit (correlation with source) | 0.247 | **0.970** |
| Expert record, 4-bit asymmetric | raised `RuntimeError` | **0.997** |
| `linear_attn.A_log` | 48 values → 6 distinct | fp16, **bit-exact** |
| `conv1d.weight` | `group_size=128`, 32× padded | `group_size=4`, 0.996 |

End-to-end through the CLI on a synthetic hybrid checkpoint: `package` → `validate
--mode full` → decode. **33 items, 0 errors, PASS.**

## Re-planned `PT-Q4E` for Qwen3.8-Flash-Next

| | before | after |
| :--- | ---: | ---: |
| dense | 2.88 GiB | 2.837 GiB |
| experts | 59.81 GiB | 59.812 GiB |
| PLE table | 24.91 GiB | 24.912 GiB |
| **total** | 87.61 GiB | **87.562 GiB** |
| fp16-pinned dense tensors | 96 (routers) | **252** (96 routers + 156 norms) |
| padded records | unknown | **0** |

The size barely moved, which is the point: correctness here costs about 24 KB.

## Still open

- **No reference forward pass exists** for the 64-layer GDN + full-attention
  hybrid. A correct package still cannot generate text. That is the FORK GATE.
- R3 has captured no real routing traces, so R4's `PROCEED_Q4E` rests on a
  synthetic Zipf trace. See [`R1c-decode-defects.md`](R1c-decode-defects.md).
- 3-bit and 6-bit still occupy 4 and 8 bits on disk (T1.9).
