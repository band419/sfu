"""
顶层仿真：将三部分串联

  输入 x (float32)
    → 前处理 (参数归约)
    → 表插值 (HOTBM sin/cos 求值)
    → 后处理 (象限重构 + 异常处理)
  输出 sin(x), cos(x) (float32)
"""

import math

from .config import HWConfig, DEFAULT_CONFIG
from .preprocessing import argument_reduce, argument_reduce_reference
from .table_interpolation import evaluate_sincos
from .postprocessing import quadrant_reconstruct, float32_round


def sincos_hw(x: float, cfg: HWConfig = DEFAULT_CONFIG,
              method: str = "table", addr_bits: int = 8,
              verbose: bool = False):
    """
    硬件行为级仿真：计算 sin(x) 和 cos(x)。

    Args:
        x:         输入浮点数
        cfg:       硬件配置
        method:    "ideal" 或 "table"
        addr_bits: HOTBM 表地址位宽 (仅 method="table" 时有效)
        verbose:   是否打印中间结果

    Returns:
        (sin_x, cos_x) 元组
    """
    # ═════════ 第一部分：前处理 (参数归约) ═════════
    ar = argument_reduce(x, cfg)

    if verbose:
        print(f"[前处理] x = {x}")
        print(f"  exn={ar.exn}, S_x={ar.S_x}, k={ar.k} (k%4={ar.k & 3})")
        print(f"  y_sign={ar.y_sign}, Y_fix=0x{ar.Y_fix:x} "
              f"({ar.Y_fix / (1 << cfg.y_fixedpoint_bits):.10f})")
        print(f"  E_y={ar.E_y}, M_y=0x{ar.M_y:x} "
              f"({ar.M_y / (1 << cfg.w_F) if ar.M_y else 0:.10f})")

        # 参考值
        if ar.exn == 1:
            k_ref, y_ref = argument_reduce_reference(x)
            print(f"  [参考] k_ref={k_ref}, y_ref={y_ref:.10f}")

    # ═════════ 第二部分：表插值 (sin/cos 求值) ═════════
    eval_result = evaluate_sincos(
        Y_fix=ar.Y_fix,
        E_y=ar.E_y,
        M_y=ar.M_y,
        y_sign=ar.y_sign,
        cfg=cfg,
        method=method,
        addr_bits=addr_bits,
    )

    if verbose and ar.exn == 1:
        y_val = ar.Y_fix / (1 << cfg.y_fixedpoint_bits)
        print(f"[表插值]")
        print(f"  sin(π/4·|y|) = {eval_result.sin_val:.10f} "
              f"(参考: {math.sin(math.pi / 4 * y_val):.10f})")
        print(f"  cos(π/4·|y|) = {eval_result.cos_val:.10f} "
              f"(参考: {math.cos(math.pi / 4 * y_val):.10f})")

    # ═════════ 第三部分：后处理 (象限重构) ═════════
    output = quadrant_reconstruct(
        sin_val=eval_result.sin_val,
        cos_val=eval_result.cos_val,
        sin_sign=eval_result.sin_sign,
        k=ar.k,
        S_x=ar.S_x,
        exn=ar.exn,
        cfg=cfg,
    )

    # 舍入到 float32
    sin_out = float32_round(output.sin_result) if output.sin_exn == 1 else output.sin_result
    cos_out = float32_round(output.cos_result) if output.cos_exn == 1 else output.cos_result

    if verbose:
        print(f"[后处理]")
        print(f"  sin({x}) = {sin_out:.10f}  (参考: {math.sin(x):.10f})")
        print(f"  cos({x}) = {cos_out:.10f}  (参考: {math.cos(x):.10f})")
        if ar.exn == 1:
            sin_err = abs(sin_out - math.sin(x))
            cos_err = abs(cos_out - math.cos(x))
            sin_ref = abs(math.sin(x))
            cos_ref = abs(math.cos(x))
            sin_rel = sin_err / sin_ref if sin_ref > 1e-30 else sin_err
            cos_rel = cos_err / cos_ref if cos_ref > 1e-30 else cos_err
            print(f"  sin 绝对误差: {sin_err:.2e}, 相对误差: {sin_rel:.2e}")
            print(f"  cos 绝对误差: {cos_err:.2e}, 相对误差: {cos_rel:.2e}")
        print()

    return sin_out, cos_out
