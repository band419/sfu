# FP16 Sin/Cos Golden Model Specification

## 1. Scope

This document defines a **bit-exact referee model** for binary16 (`IEEE-754 half`) `sin/cos`.
The model is intentionally **decoupled** from any hardware approximation path (Q-format, octant reduction, LUT/polynomial segmentation, etc.).

- Input: `mode` (`sin` or `cos`) + `x_bits` (16-bit unsigned integer)
- Output: single deterministic `y_bits` (16-bit unsigned integer)

The model computes high-precision math first, then rounds to binary16 using **roundTiesToEven**.

### Input Precondition (Reduced Value)

- The caller must provide an already **range-reduced** finite input value for the sin/cos path.
- This golden model intentionally does **not** include extra handling for non-reduced finite inputs.
- If non-reduced values are provided, behavior is out of this spec's intended usage contract.

## 2. Bit-level Contract

- `x_bits` is interpreted exactly as IEEE-754 binary16 bit pattern.
- `y_bits` is returned as IEEE-754 binary16 bit pattern.
- No payload propagation for NaN: all NaN outputs are canonical.

### Canonical qNaN

- Canonical quiet NaN bit pattern: `0x7E00`
- Applied to:
  - input NaN (any payload/signaling/quiet)
  - `sin(±inf)`
  - `cos(±inf)`

## 3. Special-value Priority

Given input `x_bits`, classify first:

1. `NaN`  -> canonical qNaN (`0x7E00`)
2. `±inf` -> canonical qNaN (`0x7E00`)
3. finite -> evaluate high-precision function and round to binary16

Additional finite identities:

- `sin(+0) = +0` => `0x0000`
- `sin(-0) = -0` => `0x8000`
- `cos(±0) = +1` => `0x3C00`

## 4. Rounding Rule

- Target format: IEEE-754 binary16
- Rounding mode: **roundTiesToEven**
- Output must be unique and deterministic for every input bit pattern.

## 5. Precision-stability Strategy

To avoid implementation-coupled behavior, golden computation uses increasing MPFR precision until the binary16-rounded result becomes stable.

Recommended progression:

- start at 128 bits
- double precision each iteration
- stop when two consecutive rounded `y_bits` match
- fail only if max precision cap is exceeded

## 6. Deliverables

- Callable Python API in `simulation/golden/`
- Directed vectors in `simulation/vectors/` with columns:
  - `mode,x_bits,y_bits,class_tag`
- Self-check tests in `simulation/tests/` validating:
  - special-value behavior
  - rerun bit-exactness
  - precision-stability
