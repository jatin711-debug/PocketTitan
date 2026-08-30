# R1c — Why `demo_chat.py` Produced Gibberish

```text
PHASE:        R1c - defect pass triggered by observed gibberish output
GATE METRIC:  logit entropy of the runner's forward pass vs log2(vocab)
              hand-rolled decode == library decode, bit-exact
DECISION:     package decode is SOUND; the runner is not a forward pass
              3 unrelated defects found and fixed; 27B package needs a rebuild
EVIDENCE:     261 offline tests (was 224 passing / 6 failing on entry)
SURPRISES:    4 (crc32c regression, non-2-D dims, zero-point clamp, norm capture)
NEXT:         decide on rebuild; implement a real forward pass or take the fork
```

## The reported symptom

`python demo_chat.py` emitted multilingual token soup at 0.1 tok/s.

## Diagnosis

Two hypotheses, tested separately.

**H1 — the package decodes wrong. Rejected.** Decoding three MLP weights two
independent ways (the hand-rolled decoder inside `demo_chat.py`, and the
library's `RTNQuantizer.dequantize` reached only through the manifest) agrees
**bit-exactly**, `max|diff| = 0`. Reconstructed weight std is 0.0102–0.0132,
which is the expected magnitude for a trained transformer. `validate --mode
fast` passes over all 851 items / 13.93 GiB with 0 errors.

**H2 — the runner is not a forward pass. Confirmed.** Measured on the prompt
`"The capital of France is"`, vocabulary 248,320 (max entropy 17.92 bits):

| Path | Logit entropy | % of max | Top token |
| :--- | ---: | ---: | :--- |
| `embed → norm → lm_head` (zero layers) | 17.21 bits | 96.0% | `oug` @ 0.0004 |
| the runner's 4-of-64-layer path | 17.30 bits | 96.5% | `' '` @ 0.0013 |

The output distribution is indistinguishable from uniform, and **running the
four layers makes it very slightly worse than running none**. Sampling that at
`temperature=0.7` produces exactly the observed soup.

The cause is in `generate()`: it iterates `[0, 1, 62, 63]` — 4 of 64 layers —
applies only `mlp.*`, and has no attention, no RoPE, and no KV cache. With no
attention there is no mechanism for one token to influence another, so the
prompt cannot affect the prediction. The gibberish is what this program is
specified to do. It is a valid load-path smoke test and nothing more.

## Defects found while diagnosing

**1. CRC32C hardware acceleration was broken** (`package/integrity.py`, from
`a824018`). `google_crc32c.Checksum` has no `.value` — the seeded call raised
`AttributeError`, failing 6 validator tests. The seeding was also wrong:
`Checksum(x)` treats `x` as leading *data*, not as a CRC seed. The seeded path
accumulates PLE shard checksums row by row, and the accelerated backend is
optional, so a package written with it installed would have failed validation on
a machine without it. Fixed with `google_crc32c.extend`, verified equal to the
pure-Python table. The existing golden vector only covered `crc == 0`.

**2. `dequantize` mis-derived dimensions for any non-2-D tensor** (all 7
quantizers). Every `quantize` flattens with `weight.view(-1, shape[-1])`; every
`dequantize` read `(shape[0], shape[1])`. Those agree only for 2-D weights. A
1-D vector came back transposed and a 3-D kernel got the wrong row width, so
`A_log`, `dt_bias`, the attention norms and `conv1d.weight` — **224 tensors in
the 27B package** — were written in a form that could not be read back. Fixed by
a single `matrix_dims` helper in `quantizers/base.py`, which `package/format.py`
now delegates to so the planner and the backends cannot drift again.

**3. The affine zero-point was clamped into the code range** (`rtn.py`,
`hqq.py` ×2). The zero-point is stored as fp16 and only ever used as
`(q - z) * s`, so it may legitimately fall outside `[0, max_int]`; a group that
does not straddle zero *requires* that. Clamping forces the representable
interval to include 0 and spends the whole budget on the empty gap. Measured on
`layers.0.linear_attn.norm.weight`: scale `0.0206`, zero `0.0`, and **all 128
codes written as `7`**, decoding to the single constant `0.1445`. The layer's
per-channel gain was erased.

Scope: 0 of 696,320 groups in `mlp.gate_proj` and `mlp.down_proj` are
non-straddling, so the 4-bit weight matrices are unaffected. The bug bites
narrow-band all-positive tensors — which is precisely what norms are.

**4. Norms and recurrence state were captured by the attention-family rules**
(`audit/classify.py`). `\.linear_attn\.` and `\.self_attn\.` are prefix rules
ordered *above* the norm rule, so they swallowed `linear_attn.norm.weight`,
`A_log`, `dt_bias`, and `self_attn.{q,k}_norm.weight` into 3- and 4-bit buckets.
This is the same shape of bug as the `shared_expert_gate` capture fixed in R0,
and PT-Q4E explicitly specifies 16 bits for exactly these tensors. Measured
damage in the built package:

| Tensor | Values | Distinct after decode |
| :--- | ---: | ---: |
| `layers.0.linear_attn.A_log` | 48 | **6** |
| `layers.0.linear_attn.dt_bias` | 48 | **8** |
| `layers.0.linear_attn.norm.weight` | 128 | **1** |
| `layers.0.input_layernorm.weight` (correctly 16-bit) | 5,120 | 1,238 |

Fixed by hoisting the norm and state rules above the family prefixes but below
`hyper_connection`, preserving the pinned `hc_norm` classification. 17,280
params move from `GDN_ATTN`/`FULL_ATTN` to `NORM` in the Flash-Next fixture; the
total is conserved exactly and the pinned constants were updated. Cost of the
fix is about 24 KB on a 14 GiB package.

## Open, not fixed

`linear_attn.conv1d.weight` has shape `(10240, 1, 4)`. At `group_size=128` its
4 input features pad to 128, so 1,966,080 params occupy **33,423,360 bytes** —
17× what fp16 would cost. This is the group-padding blowup the planner warns
about, and it is a precision-map policy question, not a bug: either pin the
tensor to fp16 or give it a group size that divides 4.

## Consequence for the built package

`qwen27b_full/` was written before defects 2–4 were fixed. Its 13.93 GiB of
weight matrices are sound and verified, but its norms and recurrence state are
destroyed and 224 tensors are unreadable. It cannot produce correct output and
must be rebuilt.

Separately: the package cannot be *run* by anything in this repository yet. The
checkpoint is a 64-layer hybrid — 48 `linear_attn` (Gated DeltaNet) plus 16
`self_attn` — and no reference forward pass for that architecture exists here.
That is the FORK GATE work, and it is the real blocker on seeing text.
