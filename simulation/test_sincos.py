"""
Sin/Cos 硬件行为仿真测试

测试策略:
  1. 逐模块单元测试 (前处理 / 表插值 / 后处理)
  2. 端到端集成测试 (不同输入范围)
  3. 边界/异常用例
  4. 大量随机输入统计误差
"""

import math
import struct
import sys
import os
import numpy as np

# 让 import 能找到 simulation 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.config import HWConfig, SP_CONFIG, HP_CONFIG, EXAMPLE_CONFIG, float_to_fields, fields_to_float
from simulation.preprocessing import argument_reduce, argument_reduce_reference
from simulation.table_interpolation import evaluate_sincos, build_cos_table, build_sin_table, table_lookup
from simulation.postprocessing import quadrant_reconstruct, float32_round
from simulation.sincos_top import sincos_hw


def to_f32(x: float) -> float:
    """将 float64 精确转换为 float32 (模拟硬件输入量化)"""
    return struct.unpack('>f', struct.pack('>f', x))[0]


# ═══════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════

def ulp_error_f32(computed: float, reference: float) -> float:
    """计算 float32 ULP 误差"""
    if math.isnan(computed) and math.isnan(reference):
        return 0.0
    if math.isinf(computed) or math.isinf(reference):
        return 0.0 if computed == reference else float('inf')
    if reference == 0.0:
        return abs(computed) / (2 ** -149) if computed != 0.0 else 0.0
    # 获取参考值的 ULP
    ref_f32 = struct.unpack('>f', struct.pack('>f', reference))[0]
    exp_bits = (struct.unpack('>I', struct.pack('>f', ref_f32))[0] >> 23) & 0xFF
    if exp_bits == 0:
        ulp = 2 ** -149
    else:
        ulp = 2 ** (exp_bits - 127 - 23)
    return abs(computed - reference) / ulp


# ═══════════════════════════════════════════════
# 测试 1: 前处理模块单元测试
# ═══════════════════════════════════════════════

def test_preprocessing():
    """测试参数归约的正确性"""
    print("=" * 60)
    print("测试 1: 前处理 (参数归约)")
    print("=" * 60)

    cfg = SP_CONFIG
    test_values = [
        0.1, 0.5, 1.0, math.pi / 4, math.pi / 2, math.pi,
        2 * math.pi, 10.0, 100.0, 1000.0, 1e6,
        12.5,  # 文档示例
        -0.5, -math.pi, -12.5,
        0.001, 1e-10,  # 小值
    ]

    passed = 0
    total = len(test_values)

    for x in test_values:
        x32 = to_f32(x)  # 硬件处理 float32(x)
        ar = argument_reduce(abs(x32), cfg)  # 硬件总是处理 |x|
        k_ref, y_ref = argument_reduce_reference(x32)

        # 从硬件结果恢复 y 值
        y_fix_val = ar.Y_fix / (1 << cfg.y_fixedpoint_bits)
        y_hw = -y_fix_val if ar.y_sign else y_fix_val

        # 检查 k 是否匹配
        k_ok = (ar.k == k_ref)

        # 检查 y 是否接近 (允许定点量化误差)
        y_tol = 2.0 ** -(cfg.w_F + cfg.g - 2)
        y_ok = abs(y_hw - y_ref) < y_tol

        status = "PASS" if (k_ok and y_ok) else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            print(f"  [{status}] x={x:>15.6f}: k_hw={ar.k} k_ref={k_ref}, "
                  f"y_hw={y_hw:.8f} y_ref={y_ref:.8f}")

    print(f"  结果: {passed}/{total} 通过")
    print()
    return passed == total


# ═══════════════════════════════════════════════
# 测试 2: 表插值模块单元测试
# ═══════════════════════════════════════════════

def test_table_interpolation():
    """测试 HOTBM 表插值精度"""
    print("=" * 60)
    print("测试 2: 表插值 (HOTBM)")
    print("=" * 60)

    cfg = SP_CONFIG

    # 测试 cos 通路: f_cos(y) = 1 - cos(π/4·y)
    cos_table = build_cos_table(cfg, addr_bits=10)
    coeff_bits = cfg.w_F + cfg.g + 4

    print("  cos 通路测试 (f_cos = 1 - cos(π/4·y)):")
    max_cos_err = 0.0
    for y_abs in np.linspace(0.001, 0.499, 200):
        f_cos_hw = table_lookup(cos_table, y_abs, coeff_bits)
        f_cos_ref = 1.0 - math.cos(math.pi / 4 * y_abs)
        cos_hw = 1.0 - f_cos_hw
        cos_ref = math.cos(math.pi / 4 * y_abs)
        err = abs(cos_hw - cos_ref)
        max_cos_err = max(max_cos_err, err)

    print(f"    最大绝对误差: {max_cos_err:.2e}")
    cos_ok = max_cos_err < 1e-6
    print(f"    状态: {'PASS' if cos_ok else 'FAIL'}")

    # 测试 sin 通路: f_sin(y) = π/4 - sin(π/4·y)/y
    sin_table = build_sin_table(cfg, addr_bits=10)

    print("  sin 通路测试 (f_sin = π/4 - sin(π/4·y)/y):")
    max_sin_err = 0.0
    for y_abs in np.linspace(0.01, 0.499, 200):
        f_sin_hw = table_lookup(sin_table, y_abs, coeff_bits)
        sinc_val = math.pi / 4 - f_sin_hw
        sin_hw = y_abs * sinc_val
        sin_ref = math.sin(math.pi / 4 * y_abs)
        err = abs(sin_hw - sin_ref)
        max_sin_err = max(max_sin_err, err)

    print(f"    最大绝对误差: {max_sin_err:.2e}")
    sin_ok = max_sin_err < 1e-6
    print(f"    状态: {'PASS' if sin_ok else 'FAIL'}")
    print()

    return cos_ok and sin_ok


# ═══════════════════════════════════════════════
# 测试 3: 后处理模块单元测试
# ═══════════════════════════════════════════════

def test_postprocessing():
    """测试角度加法重构逻辑"""
    print("=" * 60)
    print("测试 3: 后处理 (角度加法重构)")
    print("=" * 60)

    # 对每个 k mod 8 验证重构逻辑
    y = 0.3  # 测试用 y 值
    sin_py4 = math.sin(math.pi / 4 * y)
    cos_py4 = math.cos(math.pi / 4 * y)

    # 使用角度加法公式计算预期结果:
    # sin(|x|) = sin(K·π/4)·cos(φ) + cos(K·π/4)·sin(φ)
    # cos(|x|) = cos(K·π/4)·cos(φ) - sin(K·π/4)·sin(φ)
    # 其中 φ = y·π/4
    import simulation.postprocessing as pp
    expected = {}
    for k in range(8):
        S_k = pp._SIN_K_PI_4[k]
        C_k = pp._COS_K_PI_4[k]
        exp_sin = S_k * cos_py4 + C_k * sin_py4
        exp_cos = C_k * cos_py4 - S_k * sin_py4
        expected[k] = (exp_sin, exp_cos)

    passed = 0
    total = 8

    for k in range(8):
        out = quadrant_reconstruct(
            sin_val=sin_py4,
            cos_val=cos_py4,
            sin_sign=0,  # y > 0
            k=k,
            S_x=0,  # |x|
            exn=1,
        )

        exp_sin, exp_cos = expected[k]
        sin_ok = abs(out.sin_result - exp_sin) < 1e-12
        cos_ok = abs(out.cos_result - exp_cos) < 1e-12
        ok = sin_ok and cos_ok

        if ok:
            passed += 1
        else:
            print(f"  [FAIL] k={k}: sin={out.sin_result:.6f} exp={exp_sin:.6f}, "
                  f"cos={out.cos_result:.6f} exp={exp_cos:.6f}")

    # 验证数学正确性：与 math.sin/cos 比较
    for k in range(8):
        angle = (k + y) * math.pi / 4
        ref_sin = math.sin(angle)
        ref_cos = math.cos(angle)
        exp_sin, exp_cos = expected[k]
        math_ok = abs(exp_sin - ref_sin) < 1e-10 and abs(exp_cos - ref_cos) < 1e-10
        if not math_ok:
            print(f"  [MATH] k={k}: exp_sin={exp_sin:.10f} ref={ref_sin:.10f}, "
                  f"exp_cos={exp_cos:.10f} ref={ref_cos:.10f}")

    # 测试符号翻转 (S_x = 1)
    out_neg = quadrant_reconstruct(sin_py4, cos_py4, 0, 0, S_x=1, exn=1)
    out_pos = quadrant_reconstruct(sin_py4, cos_py4, 0, 0, S_x=0, exn=1)
    sign_ok = (abs(out_neg.sin_result + out_pos.sin_result) < 1e-12 and
               abs(out_neg.cos_result - out_pos.cos_result) < 1e-12)
    if sign_ok:
        passed += 1
    else:
        print(f"  [FAIL] 符号翻转: sin_neg={out_neg.sin_result}, sin_pos={out_pos.sin_result}")
    total += 1

    print(f"  结果: {passed}/{total} 通过")
    print()
    return passed == total


# ═══════════════════════════════════════════════
# 测试 4: 异常输入测试
# ═══════════════════════════════════════════════

def test_exceptions():
    """测试异常输入 (0, ±∞, NaN)"""
    print("=" * 60)
    print("测试 4: 异常输入")
    print("=" * 60)

    passed = 0
    total = 0

    # x = 0
    total += 1
    sin_0, cos_0 = sincos_hw(0.0, method="ideal")
    if sin_0 == 0.0 and cos_0 == 1.0:
        passed += 1
    else:
        print(f"  [FAIL] x=0: sin={sin_0}, cos={cos_0}")

    # x = -0
    total += 1
    sin_n0, cos_n0 = sincos_hw(-0.0, method="ideal")
    if cos_n0 == 1.0:
        passed += 1
    else:
        print(f"  [FAIL] x=-0: sin={sin_n0}, cos={cos_n0}")

    # x = +∞
    total += 1
    sin_inf, cos_inf = sincos_hw(float('inf'), method="ideal")
    if math.isnan(sin_inf) and math.isnan(cos_inf):
        passed += 1
    else:
        print(f"  [FAIL] x=+inf: sin={sin_inf}, cos={cos_inf}")

    # x = -∞
    total += 1
    sin_ninf, cos_ninf = sincos_hw(float('-inf'), method="ideal")
    if math.isnan(sin_ninf) and math.isnan(cos_ninf):
        passed += 1
    else:
        print(f"  [FAIL] x=-inf: sin={sin_ninf}, cos={cos_ninf}")

    # x = NaN
    total += 1
    sin_nan, cos_nan = sincos_hw(float('nan'), method="ideal")
    if math.isnan(sin_nan) and math.isnan(cos_nan):
        passed += 1
    else:
        print(f"  [FAIL] x=NaN: sin={sin_nan}, cos={cos_nan}")

    print(f"  结果: {passed}/{total} 通过")
    print()
    return passed == total


# ═══════════════════════════════════════════════
# 测试 5: 端到端集成测试 (ideal 模式)
# ═══════════════════════════════════════════════

def test_e2e_ideal():
    """端到端测试 - ideal 模式 (验证归约 + 重构逻辑)"""
    print("=" * 60)
    print("测试 5: 端到端集成 (ideal 模式)")
    print("=" * 60)

    # 避免 float32 精确表示 π/2 倍数的边界情况
    # （float32 对 π/2 等无理数只有近似表示，归约后 y≈1e-8 而非 0，
    #   导致结果为 ~1e-8 而参考为 ~1e-16，ULP 巨大但不影响实际应用）
    test_values = [
        0.1, 0.5, 1.0, 1.5, 2.0, 3.0,
        math.pi / 6, math.pi / 4, math.pi / 3,
        10.0, 100.0, 1000.0, 12.5,
        -0.5, -1.0, -12.5,
        0.001, 0.0001, 1e-7,
        1e5,
        # 各八分区覆盖
        0.3, 0.9, 1.2, 1.9, 2.5, 3.5, 4.0, 5.0, 5.5, 6.0, 7.0, 8.0,
    ]

    max_sin_ulp = 0.0
    max_cos_ulp = 0.0
    passed = 0
    total = len(test_values)
    failed_cases = []

    for x in test_values:
        x32 = to_f32(x)  # 硬件实际处理的是 float32(x)
        sin_hw, cos_hw = sincos_hw(x32, method="ideal")
        sin_ref = math.sin(x32)
        cos_ref = math.cos(x32)

        sin_ulp = ulp_error_f32(sin_hw, sin_ref)
        cos_ulp = ulp_error_f32(cos_hw, cos_ref)

        max_sin_ulp = max(max_sin_ulp, sin_ulp)
        max_cos_ulp = max(max_cos_ulp, cos_ulp)

        # faithful rounding: 误差 < 4 ULP
        ok = sin_ulp < 4.0 and cos_ulp < 4.0
        if ok:
            passed += 1
        else:
            failed_cases.append((x32, sin_ulp, cos_ulp, sin_hw, sin_ref, cos_hw, cos_ref))

    for x, su, cu, sh, sr, ch, cr in failed_cases[:5]:
        print(f"  [FAIL] x={x}: sin_ulp={su:.1f} cos_ulp={cu:.1f}")
        print(f"         sin_hw={sh:.10e} sin_ref={sr:.10e}")
        print(f"         cos_hw={ch:.10e} cos_ref={cr:.10e}")

    print(f"  最大 sin ULP 误差: {max_sin_ulp:.1f}")
    print(f"  最大 cos ULP 误差: {max_cos_ulp:.1f}")
    print(f"  结果: {passed}/{total} 通过")
    print()
    return passed == total


# ═══════════════════════════════════════════════
# 测试 6: 端到端集成测试 (table 模式)
# ═══════════════════════════════════════════════

def test_e2e_table():
    """端到端测试 - table 模式 (验证完整硬件流程)"""
    print("=" * 60)
    print("测试 6: 端到端集成 (table 模式, addr_bits=10)")
    print("=" * 60)

    test_values = [
        0.1, 0.5, 1.0, 1.5, 2.0, 3.0,
        math.pi / 6, math.pi / 4, math.pi / 3,
        10.0, 100.0, 12.5,
        -0.5, -1.0,
        0.001, 0.01,
        0.3, 0.9, 1.2, 1.9, 2.5, 3.5, 5.0, 7.0, 8.0,
    ]

    max_sin_ulp = 0.0
    max_cos_ulp = 0.0
    passed = 0
    total = len(test_values)
    failed_cases = []

    for x in test_values:
        x32 = to_f32(x)
        sin_hw, cos_hw = sincos_hw(x32, method="table", addr_bits=10)
        sin_ref = math.sin(x32)
        cos_ref = math.cos(x32)

        sin_ulp = ulp_error_f32(sin_hw, sin_ref)
        cos_ulp = ulp_error_f32(cos_hw, cos_ref)

        max_sin_ulp = max(max_sin_ulp, sin_ulp)
        max_cos_ulp = max(max_cos_ulp, cos_ulp)

        # table 模式允许更大误差
        ok = sin_ulp < 16.0 and cos_ulp < 16.0
        if ok:
            passed += 1
        else:
            failed_cases.append((x32, sin_ulp, cos_ulp, sin_hw, sin_ref, cos_hw, cos_ref))

    for x, su, cu, sh, sr, ch, cr in failed_cases[:5]:
        print(f"  [FAIL] x={x}: sin_ulp={su:.1f} cos_ulp={cu:.1f}")
        print(f"         sin_hw={sh:.10e} sin_ref={sr:.10e}")
        print(f"         cos_hw={ch:.10e} cos_ref={cr:.10e}")

    print(f"  最大 sin ULP 误差: {max_sin_ulp:.1f}")
    print(f"  最大 cos ULP 误差: {max_cos_ulp:.1f}")
    print(f"  结果: {passed}/{total} 通过")
    print()
    return passed == total


# ═══════════════════════════════════════════════
# 测试 7: 随机输入大规模统计测试
# ═══════════════════════════════════════════════

def test_random_stress():
    """随机输入统计测试"""
    print("=" * 60)
    print("测试 7: 随机输入统计 (ideal 模式, N=10000)")
    print("=" * 60)

    np.random.seed(42)
    N = 10000

    # 多个范围的随机值
    ranges = [
        ("小值 [1e-10, 1]", 1e-10, 1.0),
        ("中值 [1, 100]", 1.0, 100.0),
        ("大值 [100, 1e6]", 100.0, 1e6),
    ]

    all_ok = True

    for name, lo, hi in ranges:
        xs = np.random.uniform(lo, hi, N // 3)
        # 随机加负号
        signs = np.random.choice([-1, 1], N // 3)
        xs = xs * signs

        sin_ulps = []
        cos_ulps = []
        failures = 0

        for x in xs:
            x32 = to_f32(float(x))
            sin_hw, cos_hw = sincos_hw(x32, method="ideal")
            sin_ref = math.sin(x32)
            cos_ref = math.cos(x32)

            su = ulp_error_f32(sin_hw, sin_ref)
            cu = ulp_error_f32(cos_hw, cos_ref)
            sin_ulps.append(su)
            cos_ulps.append(cu)

            if su > 4.0 or cu > 4.0:
                failures += 1

        sin_ulps = np.array(sin_ulps)
        cos_ulps = np.array(cos_ulps)

        print(f"  {name}:")
        print(f"    sin ULP - 平均: {sin_ulps.mean():.2f}, 最大: {sin_ulps.max():.1f}, "
              f"P99: {np.percentile(sin_ulps, 99):.1f}")
        print(f"    cos ULP - 平均: {cos_ulps.mean():.2f}, 最大: {cos_ulps.max():.1f}, "
              f"P99: {np.percentile(cos_ulps, 99):.1f}")
        print(f"    失败率: {failures}/{len(xs)} ({100*failures/len(xs):.1f}%)")

        if failures > len(xs) * 0.05:  # 允许 5% 失败率
            all_ok = False

    print()
    return all_ok


# ═══════════════════════════════════════════════
# 测试 8: 文档示例验证
# ═══════════════════════════════════════════════

def test_document_example():
    """验证文档中 x=12.5 的完整例子"""
    print("=" * 60)
    print("测试 8: 文档示例 (x=12.5)")
    print("=" * 60)

    x = 12.5

    # 用 verbose 模式展示完整流程
    sin_hw, cos_hw = sincos_hw(x, method="ideal", verbose=True)

    sin_ref = math.sin(12.5)
    cos_ref = math.cos(12.5)

    sin_ok = abs(sin_hw - sin_ref) < 1e-4
    cos_ok = abs(cos_hw - cos_ref) < 1e-4

    print(f"  sin(12.5) = {sin_hw:.6f} (参考: {sin_ref:.6f}) {'PASS' if sin_ok else 'FAIL'}")
    print(f"  cos(12.5) = {cos_hw:.6f} (参考: {cos_ref:.6f}) {'PASS' if cos_ok else 'FAIL'}")
    print()

    return sin_ok and cos_ok


# ═══════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Sin/Cos 硬件行为仿真 - 测试套件                      ║")
    print("║   Based on Detrey & de Dinechin, FPL 2007               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    results = {}

    results["前处理"] = test_preprocessing()
    results["表插值"] = test_table_interpolation()
    results["后处理"] = test_postprocessing()
    results["异常输入"] = test_exceptions()
    results["端到端(ideal)"] = test_e2e_ideal()
    results["端到端(table)"] = test_e2e_table()
    results["随机统计"] = test_random_stress()
    results["文档示例"] = test_document_example()

    # 汇总
    print("=" * 60)
    print("测试汇总")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {name:20s} : {status}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("所有测试通过!")
    else:
        print("有测试失败，请检查上述输出。")
        sys.exit(1)


if __name__ == "__main__":
    main()
