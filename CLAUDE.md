# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Sine/Cosine Floating-Point Unit (SFU) simulator** — a Python framework for evaluating IEEE-754 binary16 (FP16) sin/cos hardware approximation strategies. The goal is to measure ULP (Unit in the Last Place) error of different hardware datapath designs against a bit-exact golden reference.

## Running Tests

```bash
# Run all unit tests
python -m pytest simulation/tests/ -v

# Run a specific test file
python -m pytest simulation/tests/test_golden_fp16.py -v
```

## Main Workflow

```bash
# Edit config first: simulation/ulp_compare_config.py
# Then run the batch ULP comparison:
python simulation/run_ulp_compare.py
# Output: simulation/vectors/ulp_compare_summary.csv
```

## Compare Driver (single-vector comparison)

```bash
python simulation/compare.py --list-duts
python simulation/compare.py --dut qflow-ph --vectors simulation/vectors/directed_fp16_sincos.csv
```

## Dependencies

- `gmpy2` (required — arbitrary-precision arithmetic for golden model and DUT kernels)
- No `requirements.txt` exists; install manually: `pip install gmpy2`

## Architecture

### Golden Model (`simulation/golden/model.py`)

Bit-exact FP16 sin/cos referee. Input contract: **caller must supply range-reduced input** (x already in [0, 2π)); the golden model does not perform range reduction itself. Uses adaptive-precision mpfr starting at 128 bits, doubling until the result stabilizes. Special values (NaN, inf) return canonical qNaN `0x7E00`.

API: `golden_fp16(mode: "sin"|"cos", x_bits: int) -> int`

### Three DUT Families

All DUTs implement the same **Q-scheme pipeline**:

```
|x| → reduction → (k mod 4, y) → dual kernel {s(y), c(y)} → MUX(mode, q) → sign fix → FP16
```

| DUT | Name pattern | Reduction | Kernel |
|-----|-------------|-----------|--------|
| 1 | `qflow-fixedconst-{fp16,fp32,fp128}` | Fixed-precision 2/π multiply | converging mpfr |
| 2 | `qflow-ph` | Payne-Hanek sliding window | converging mpfr |
| 3 | `qflow-ph-interp-{u,nu}{16..256}-{lin,quad}` | Payne-Hanek sliding window | LUT + interpolation |

DUT 1 isolates reduction error from kernel error. DUT 2 validates PH accuracy without high-precision FP in the datapath. DUT 3 models realistic hardware: FP32 LUT + linear or quadratic Lagrange interpolation, with uniform (`u`) or non-uniform power (`nu`, t=u²) segment partitioning.

### Pluggable DUT Adapter System

- `simulation/dut/base.py` — `DUTAdapter` protocol and `DUTResult` dataclass
- `simulation/dut/registry.py` — `register_dut`, `get_dut`, `list_duts`
- `simulation/dut/adapters.py` — triggers auto-registration of all built-in adapters (must be imported)

To add a new DUT: implement `eval(mode, x_bits) -> DUTResult`, call `register_dut()`, and it becomes available by name.

### ULP Comparison Framework (`simulation/ulp_compare_models.py`)

Fixed topology: one golden vs. N models. Key API:
- `run_stats_from_config(config)` — executes batch comparison
- `summarize_stats(stats, inputs_count=...)` — generates summary rows
- `build_csv_table(rows)` — formats as CSV

Configuration lives in `simulation/ulp_compare_config.py`: set `GOLDEN`, `MODELS`, `EXHAUSTIVE` (all 65536 FP16 inputs) or `SAMPLES`/`SEED` (random subset).

### Key Shared Modules

- `simulation/dut/qflow_common.py` — quadrant selection MUX and sign reconstruction (shared by all Q-flow DUTs)
- `simulation/dut/ph_reduction.py` — Payne-Hanek window extraction
- `simulation/common/fp16.py` — FP16 bit manipulation and classification (`FP16Class` enum, `decode_fp16_bits`, `encode_fp16_bits`)
- `simulation/common/vector_gen.py` — directed and random test vector generation

## Working with Bit Patterns

All APIs use integer bit patterns (e.g., `0x3C00` for 1.0 in FP16), not Python floats. The `common/fp16.py` module provides conversion and classification utilities.

## Documentation

- `doc/SinCos_Derivation.md` — complete mathematical derivation of the Q-scheme
- `doc/Golden_Model_Spec.md` — formal spec for the golden model's special-value policy
- `simulation/README.md` — API contracts and DUT integration guide
