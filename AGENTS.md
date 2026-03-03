# AGENTS.md

Guidance for AI coding agents operating in this repository.

## Project Overview

Sine/Cosine Floating-Point Unit (SFU) simulator — Python framework for evaluating IEEE-754 binary16 (FP16) sin/cos hardware approximation strategies. Measures ULP error of hardware datapath designs against a bit-exact golden reference.

**All APIs use integer bit patterns** (e.g., `0x3C00` for 1.0 in FP16), never Python floats. Use `simulation/common/fp16.py` for conversion.

## Build & Test Commands

```bash
# Run all tests
python -m pytest simulation/tests/ -v

# Run a single test file
python -m pytest simulation/tests/test_golden_fp16.py -v

# Run a single test class
python -m pytest simulation/tests/test_ph_reduction.py::Stage2PHReduceTests -v

# Run a single test method
python -m pytest simulation/tests/test_ph_reduction.py::Stage2PHReduceTests::test_one_sin -v

# Batch ULP comparison (edit config first: simulation/ulp_compare_config.py)
python simulation/run_ulp_compare.py

# Compare driver (single DUT against golden vectors)
python simulation/compare.py --list-duts
python simulation/compare.py --dut qflow-ph --vectors simulation/vectors/directed_fp16_sincos.csv
```

**Dependency**: `gmpy2` (arbitrary-precision arithmetic). Install: `pip install gmpy2`. No requirements.txt.

## Project Structure

```
simulation/
├── __init__.py
├── common/
│   ├── fp16.py              # FP16 bit manipulation, FP16Class enum, encode/decode
│   └── vector_gen.py        # Directed/random test vector generation
├── golden/
│   └── model.py             # Bit-exact golden referee: golden_fp16(mode, x_bits)
├── dut/
│   ├── base.py              # DUTAdapter protocol + DUTResult dataclass
│   ├── registry.py          # register_dut / get_dut / list_duts
│   ├── adapters.py          # Auto-registration entry point (import triggers registration)
│   ├── qflow_common.py      # Shared quadrant MUX + sign reconstruction
│   ├── ph_reduction.py      # Payne-Hanek sliding window reduction
│   ├── qflow_fixedconst.py  # DUT 1: fixed-precision 2/π multiply
│   ├── qflow_ph.py          # DUT 2: PH reduction + converging mpfr kernel
│   └── qflow_ph_interp.py   # DUT 3: PH reduction + LUT interpolation
├── tests/
│   ├── test_golden_fp16.py
│   ├── test_fp16_pack.py
│   ├── test_qflow_common.py
│   ├── test_ph_reduction.py
│   ├── test_dut_compare_interface.py
│   └── test_ulp_compare_models.py
├── compare.py               # CLI compare driver
├── ulp_sweep.py             # ULP sweep CLI
├── ulp_compare_models.py    # Batch comparison framework
└── ulp_compare_config.py    # Comparison configuration
```

## Architecture: DUT Adapter System

All DUTs implement the Q-scheme pipeline:
```
|x| → reduction → (k mod 4, y) → dual kernel {s(y), c(y)} → MUX(mode, q) → sign fix → FP16
```

### Adding a New DUT

1. Implement the `DUTAdapter` protocol from `simulation/dut/base.py`:
   ```python
   @dataclass
   class MyDUT:
       name: str = "my-dut"
       def eval(self, mode: Mode, x_bits: int) -> DUTResult:
           ...
   ```
2. Register in `simulation/dut/adapters.py`: `register_dut(MyDUT())`
3. Import `simulation.dut.adapters` to trigger auto-registration (already done in compare/sweep scripts via `import simulation.dut.adapters  # noqa: F401`)

### Golden Model Contract

`golden_fp16(mode, x_bits)` — caller must supply range-reduced input. The model does NOT perform range reduction. Special values: NaN/Inf → `0x7E00` (canonical qNaN), sin(±0) = ±0, cos(±0) = +1.

## Code Style

### Imports

Every file starts with `from __future__ import annotations`. Import order:

1. `from __future__ import annotations` (always first)
2. Standard library (`math`, `struct`, `csv`, `argparse`, etc.)
3. Third-party (`gmpy2` — with try/except guard)
4. Local absolute (`from simulation.common.fp16 import ...`)
5. Local relative (`from .base import Mode`)

Blank line between each group. No `isort`, `black`, or `ruff` config exists.

### gmpy2 Import Guard Pattern

gmpy2 is optional at import time. Use this pattern:
```python
try:
    import gmpy2
except ImportError as exc:  # pragma: no cover
    gmpy2 = None
    _gmpy2_import_error = exc
else:
    _gmpy2_import_error = None
```

Tests skip with `@unittest.skipIf(gmpy2 is None, "gmpy2 is not installed")`.

### Type Annotations

- Used consistently on function signatures: `def foo(x: int, mode: Mode) -> int:`
- `Mode = Literal["sin", "cos"]` defined in `simulation/dut/base.py` and imported
- Modern union syntax: `list[str]`, `dict[str, str]`, `int | None` (enabled by `from __future__ import annotations`)
- Protocol classes for interfaces (not ABCs): `class DUTAdapter(Protocol):`
- `@dataclass(frozen=True)` for immutable value objects, plain `@dataclass` for mutable

### Naming Conventions

- **Functions/variables**: `snake_case` — `golden_fp16`, `x_bits`, `y_fixed`
- **Classes**: `PascalCase` — `DUTResult`, `FP16Decoded`, `QFlowPHDUT`
- **Constants**: `UPPER_SNAKE` — `CANONICAL_QNAN_BITS`, `ROM_DEPTH`, `TWO_OVER_PI_ROM`
- **Private helpers**: `_leading_underscore` — `_mode_bit`, `_extract_window`
- **Bit-pattern variables**: `x_bits`, `y_bits`, `g_bits`, `d_bits` (always `_bits` suffix)
- **Spec references in names**: constants like `W_F`, `G_K`, `N_BP` match the spec document

### Docstrings

- Imperative mood, concise: `"""Return correctly-rounded binary16 sin/cos result bits."""`
- Cross-reference spec sections: `(spec §9.3.1)`, `(SinCos_Derivation §6-§7)`
- Module-level docstrings describe purpose and spec alignment
- No specific docstring format (Google/NumPy/reST) — just plain text with inline references

### Error Handling

- `ValueError` for invalid inputs: `raise ValueError(f"mode must be 'sin' or 'cos', got {mode!r}")`
- `RuntimeError` for convergence failures: `raise RuntimeError(f"golden_fp16 did not stabilize...")`
- `KeyError` with descriptive message for registry lookups
- `TypeError` for wrong argument types: `raise TypeError(f"fp16 bits must be int, got {type(bits)!r}")`
- No custom exception classes — uses built-in exceptions throughout
- Input validation at module boundaries, not deep internals

### Tests

- Framework: `unittest.TestCase` (not pytest fixtures)
- File naming: `test_*.py` in `simulation/tests/`
- Class naming: descriptive `PascalCase` — `GoldenFp16Tests`, `Stage1UnpackTests`
- Method naming: `test_<what>` — `test_special_values_policy`, `test_y_fixed_range`
- No conftest.py, no pytest fixtures, no parametrize — plain unittest style
- Hex literals for FP16 bit patterns: `self.assertEqual(golden_fp16("sin", 0x0000), 0x0000)`
- Spec section references in test class docstrings: `"""Tests for spec-aligned Payne-Hanek reduction (§6-§7)."""`

### Formatting

- No formatter configured (no black, ruff, autopep8)
- Line length: ~100-120 chars (no hard limit, but lines rarely exceed 120)
- Trailing commas in multi-line structures
- `# noqa: F401` for intentional side-effect imports
- `# pragma: no cover` for platform-specific unreachable code
- Comment style: inline `#` comments for spec references and rationale

### Module Exports

`__init__.py` files use explicit `__all__` lists and re-export key symbols:
```python
from .base import DUTAdapter, DUTResult
from .registry import get_dut, list_duts, register_dut
__all__ = ["DUTAdapter", "DUTResult", "register_dut", "get_dut", "list_duts"]
```

## Key References

- `doc/SinCos_Derivation.md` — complete mathematical derivation of the Q-scheme
- `doc/Golden_Model_Spec.md` — formal spec for golden model special-value policy
- `simulation/README.md` — API contracts, DUT integration guide, and failure reproduction flow
