"""
硬件参数配置

可切换不同精度配置来模拟不同位宽的硬件实现。
"""

import struct
from dataclasses import dataclass


@dataclass
class HWConfig:
    """硬件位宽参数"""
    w_E: int        # 指数位宽
    w_F: int        # 尾数小数部分位宽
    g: int          # 舍入保护位
    g_K: int        # Kahan 保护位

    @property
    def E0(self) -> int:
        """指数偏移 (bias)"""
        return (1 << (self.w_E - 1)) - 1

    @property
    def window_width(self) -> int:
        """Payne-Hanek 窗口宽度 W = 2*w_F + g + g_K + 3"""
        return 2 * self.w_F + self.g + self.g_K + 3

    @property
    def y_frac_bits(self) -> int:
        """y 的小数部分位宽 = w_F + g + g_K"""
        return self.w_F + self.g + self.g_K

    @property
    def y_fixedpoint_bits(self) -> int:
        """给 cos 通路使用的定点 Y 位宽 = w_F + g"""
        return self.w_F + self.g


# 预定义配置
# 单精度 (float32)
SP_CONFIG = HWConfig(w_E=8, w_F=23, g=2, g_K=23)

# 半精度 (float16) — 用于快速验证
HP_CONFIG = HWConfig(w_E=5, w_F=10, g=2, g_K=10)

# 文档示例中的简化配置
EXAMPLE_CONFIG = HWConfig(w_E=8, w_F=4, g=2, g_K=4)

# 默认使用单精度
DEFAULT_CONFIG = SP_CONFIG


def float_to_fields(x: float, cfg: HWConfig = DEFAULT_CONFIG):
    """
    将 Python float 拆解为浮点字段 (exn, S, E, F)。
    仅支持 w_E=8, w_F=23 的 IEEE 754 单精度。

    Returns:
        exn: 2-bit 异常编码 (0=zero, 1=normal, 2=inf, 3=nan)
        S:   符号位
        E:   存储指数 (biased)
        F:   尾数小数部分 (整数, w_F 位)
    """
    assert cfg.w_E == 8 and cfg.w_F == 23, "float_to_fields 目前仅支持单精度"

    # 用 struct 将 float 打包再拆为 uint32
    bits = struct.unpack('>I', struct.pack('>f', float(x)))[0]

    S = (bits >> 31) & 1
    E = (bits >> 23) & 0xFF
    F = bits & 0x7FFFFF

    if E == 0 and F == 0:
        exn = 0  # zero
    elif E == 0xFF and F == 0:
        exn = 2  # inf
    elif E == 0xFF:
        exn = 3  # nan
    else:
        exn = 1  # normal

    return exn, S, E, F


def fields_to_float(exn: int, S: int, E: int, F: int, cfg: HWConfig = DEFAULT_CONFIG) -> float:
    """
    将浮点字段组合回 Python float。
    仅支持单精度。
    """
    assert cfg.w_E == 8 and cfg.w_F == 23, "fields_to_float 目前仅支持单精度"

    if exn == 0:
        return -0.0 if S else 0.0
    if exn == 2:
        return float('-inf') if S else float('inf')
    if exn == 3:
        return float('nan')

    bits = (S << 31) | ((E & 0xFF) << 23) | (F & 0x7FFFFF)
    return struct.unpack('>f', struct.pack('>I', bits))[0]
