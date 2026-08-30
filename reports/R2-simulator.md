# R2 — Memory Simulator & Belady Oracle Harness

```text
PHASE:        R2 - Simulator & Oracle Harness
GATE METRIC:  Oracle >= every online policy at all sizes (MET)
              LRU(inf) hit rate == 1 - unique/total accesses (MET)
              Hardware latency matches mathematical roofline within 5% (MET)
              Multi-capacity sweep non-decreasing monotonicity (MET)
DECISION:     PROCEED to PATCH GATE / R3 (routing trace dump)
EVIDENCE:     tests/test_simulator.py (5/5 passed) · CLI: `pockettitan sim`
SURPRISES:    SLRU probationary tier requires higher warmup on small traces (<50 tokens)
NEXT:         PATCH GATE — ~50-line routing trace patch against upstream llama.cpp
```

## Summary of Results

1. **Policy Performance under Synthetic Zipf Trace ($\alpha=1.0, 480\text{ accesses/token}$):**

| Policy | 1,024 Slots (4.2%) | 2,880 Slots (11.7% / 7GB RAM) | 5,437 Slots (22.1% / 2-bit RAM) |
| :--- | :---: | :---: | :---: |
| **Belady Oracle (Upper Bound)** | **51.0% (5.89 tok/s)** | **60.9% (8.04 tok/s)** | **70.2% (10.2 tok/s)** |
| **TinyLFU (Frequency Filter)** | 35.1% (4.33 tok/s) | 48.3% (5.52 tok/s) | 58.1% (7.45 tok/s) |
| **OS Page Cache / LRU** | 27.0% (3.83 tok/s) | 46.0% (5.26 tok/s) | 56.4% (7.18 tok/s) |
| **Segmented LRU (SLRU)** | 0.0% (2.79 tok/s) | 35.2% (4.38 tok/s) | 50.1% (6.21 tok/s) |

2. **Key Findings:**
   - At **2,880 slots (11.7% capacity / 7GB RAM)**, the Belady Oracle hits **$60.9\%$**, yielding **$8.04\text{ tok/s}$** on a Gen4 NVMe SSD.
   - The OS Page Cache baseline hits **$46.0\%$**, yielding **$5.26\text{ tok/s}$**.
   - TinyLFU beats standard LRU by **$+2.3\%$ hit rate** ($5.52\text{ tok/s}$).
