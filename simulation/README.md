# Simulation Golden Model

This folder provides a bit-exact FP16 `sin/cos` golden referee model.

## API

Primary API:

- `simulation.golden.model.golden_fp16(mode, x_bits) -> y_bits`

Arguments:

- `mode`: `"sin"` or `"cos"`
- `x_bits`: 16-bit integer interpreted as IEEE-754 binary16 bit pattern

Return:

- `y_bits`: 16-bit integer, uniquely rounded with binary16 `roundTiesToEven`

Notes:

- The golden model accepts any FP16 bit pattern, including special values and subnormals (FTZ).
- No range reduction is required by the caller.

Special policy is frozen in `doc/Golden_Model_Spec.md`.

## Vector format

Generated CSV columns:

- `mode`
- `x_bits` (hex, e.g. `0x3C00`)
- `y_bits` (hex, e.g. `0x3A98`)
- `class_tag` (`zero/subnormal/normal/inf/nan`)

## Usage

- Generate directed vectors by running module `simulation.golden.vector_gen`.
- Consume `simulation/vectors/directed_fp16_sincos.csv` from testbench for compare.

## Multi-DUT integration contract

To compare multiple DUT models against golden, use the pluggable adapter interface in `simulation/dut/`.

### Required adapter shape

Each DUT adapter must expose:

- `name: str` (unique registry key)
- `eval(mode, x_bits) -> DUTResult`

Where `DUTResult` contains:

- `y_bits: int` (16-bit IEEE-754 binary16 bit pattern)
- optional `meta: dict[str, str]`

Reference files:

- `simulation/dut/base.py` (`DUTAdapter`, `DUTResult`)
- `simulation/dut/registry.py` (`register_dut`, `get_dut`, `list_duts`)
- `simulation/dut/adapters.py` (example built-in adapter: `golden-mirror`)

### How to mount a new DUT

1. Add a new adapter class in `simulation/dut/adapters.py` or a sibling module.
2. Implement `eval(mode, x_bits)` following the eval interface.
3. Register it once via `register_dut(MyDUT())`.
4. Run compare driver with `--dut <name>`.

### Compare driver

Use `simulation/compare.py` to run vector-by-vector comparison:

- loads CSV vectors
- recomputes golden for each row
- calls selected DUT adapter
- reports mismatch summary and sample failing rows

Available CLI flags include:

- `--list-duts`
- `--dut <adapter-name>`
- `--vectors <csv-path>`
- `--max-report <N>`

### Interpolation Q-flow variants for horizontal comparison

Built-in DUT adapters are auto-registered and can be listed via `--list-duts`.

Three DUT families model the sin/cos datapath at different levels of fidelity:

#### DUT 1 — Fixed-format constant reduction (`qflow-fixedconst-*`)

Uses a 2/π constant stored in a fixed floating-point format and does the
reduction multiply at that precision.  Available formats: `fp16`, `fp32`, `fp128`.

- `qflow-fixedconst-fp16`
- `qflow-fixedconst-fp32`
- `qflow-fixedconst-fp128`

The sin kernel still uses converging mpfr, so **only the reduction error** is
visible.

#### DUT 2 — PH effective-window reduction (`qflow-ph`)

Uses a Payne–Hanek style sliding-window extraction of 2/π bits aligned to the
input exponent.  The sin kernel uses converging mpfr.

- `qflow-ph`

DUT 1 and DUT 2 together demonstrate whether the PH method can guarantee ULP
accuracy without introducing high-precision floating-point arithmetic in the
reduction.

#### DUT 3 — PH reduction + hardware interpolation (`qflow-ph-interp-*`)

Same PH reduction as DUT 2, but the sin kernel is replaced by a hardware-style
lookup + interpolation pipeline with configurable parameters:

- segment count: 64 / 128 / 256
- partition strategy:
  - `u` = uniform partition on $t\in[0,1]$
  - `nu` = non-uniform power partition ($t=u^{2}$)
- interpolation order:
  - `lin` = piecewise linear interpolation
  - `quad` = piecewise quadratic interpolation (3-point Lagrange)

Name format:

- `qflow-ph-interp-<partition><segments>-<order>`

Examples:

- `qflow-ph-interp-u64-lin`
- `qflow-ph-interp-u256-lin`
- `qflow-ph-interp-nu64-quad`

These DUTs keep the same PH reduction/sign framework and only swap the core
`sin(pi/2*t)` evaluation with interpolation, so you can compare interpolation
strategies directly.

For hardware-consistent modeling, interpolation variants use a numerical path of:

- `Qy` (from PH reduction) converted to FP32
- FP32 LUT nodes/values
- FP32 interpolation arithmetic (linear/quadratic)
- final cast/round to FP16 output

## Failure reproduction flow

When mismatch occurs (`dut_bits != y_bits`):

1. Record tuple: `mode`, `x_bits`, `dut_bits`, `golden_bits`.
2. Re-run single point via `golden_fp16(mode, x_bits)`.
3. Attach tuple + seed/case-id into regression log for deterministic replay.

## ULP对比框架（单一golden）

`simulation/ulp_compare_models.py` 的比较关系固定为：

- 一个 `golden`
- 多个 `model`

即所有 model 都和同一个 golden 比较，不做多 golden 矩阵对比。

常用配置参数：

- `CompareConfig.golden`：唯一 golden（默认 `golden-fp16`）
- `CompareConfig.models`：待比较模型列表
- `CompareConfig.exhaustive` 或 `CompareConfig.samples/seed`：输入集合配置
- `run_stats_from_config(config)`：按配置执行对比
- `summarize_stats(stats, inputs_count=...)`：生成汇总行
- `build_csv_table(rows)`：把汇总结果转成 CSV 表格字符串
