"""
第一部分：前处理 —— 参数归约 (Argument Reduction)

实现 Markstein 变体 + Payne-Hanek 算法：
  1. 用 Payne-Hanek 从预存 4/π 中截取与 x 的指数匹配的窗口
  2. 定点乘法：1.Fx × C
  3. 从乘积中提取 k (mod 8, 3 bits) 和 y ∈ [-1/2, 1/2]
  4. 输出 y 的两种格式：定点 Y (给 cos) 和浮点 (Ey, My) (给 sin)
"""

import math
from mpmath import mp, mpf, pi, power, floor, nint

from .config import HWConfig, DEFAULT_CONFIG, float_to_fields


# ─────────────────────────────────────────────
# 4/π 高精度常量预计算
# ─────────────────────────────────────────────

def precompute_four_over_pi(total_bits: int) -> int:
    """
    预计算 4/π 的二进制表示，返回定点整数。
    结果满足  value = result * 2^{-(total_bits - 2)}
    即小数点在第 1 位和第 0 位之间 (4/π ≈ 1.27...)，
    总共存储 total_bits 个二进制位（含整数位 "1"）。
    """
    mp.prec = total_bits + 64  # 多留一些精度
    val = mpf(4) / pi
    # 左移使之成为整数
    int_val = int(val * power(2, total_bits - 2))
    return int_val


def _get_four_over_pi_bits(total_bits: int) -> int:
    """缓存版本的预计算"""
    if not hasattr(_get_four_over_pi_bits, '_cache'):
        _get_four_over_pi_bits._cache = {}
    if total_bits not in _get_four_over_pi_bits._cache:
        _get_four_over_pi_bits._cache[total_bits] = precompute_four_over_pi(total_bits)
    return _get_four_over_pi_bits._cache[total_bits]


# ─────────────────────────────────────────────
# Payne-Hanek 窗口提取
# ─────────────────────────────────────────────

def extract_window(E: int, cfg: HWConfig) -> int:
    """
    从预存 4/π 中提取与指数 E = Ex - E0 匹配的窗口。

    窗口覆盖 4/π 的第 i_high 位到第 i_low 位:
        i_high = w_F - E + 2
        i_low  = -(E + w_F + g + g_K)

    窗口宽度 W = 2*w_F + g + g_K + 3，与 E 无关。

    返回: W 位的定点整数 (MSB 对应 i_high, LSB 对应 i_low)
    """
    w_F = cfg.w_F
    g = cfg.g
    g_K = cfg.g_K
    W = cfg.window_width

    i_high = w_F - E + 2
    i_low = -(E + w_F + g + g_K)

    # 需要 4/π 从最高位到 i_low 的所有位
    # 4/π 的整数位就是位号 0 (值=1)，位号 -1 开始是小数位
    # 我们存储时，bit index 1 对应 2^1 = 2 的系数（为0），bit index 0 对应 2^0 的系数（为1）
    # 总存储至少需要覆盖到 i_low

    # 所需的 4/π 精度：从位号 1 到 i_low，总共 2 - i_low 位
    needed_bits = 2 - i_low + 16  # 多取一些

    four_over_pi_int = _get_four_over_pi_bits(needed_bits)

    # four_over_pi_int 的最高位对应 2^1 位号，最低位对应 2^{-(needed_bits-2)} 位号
    # 位号 b 对应 four_over_pi_int 中从 MSB 起第 (1 - b) 位
    # 即第 (1 - b) 位 from MSB = 第 (needed_bits - 1 - (1 - b)) = (needed_bits - 2 + b) 位 from LSB

    # 提取 i_high 到 i_low 的窗口：
    # i_high from LSB = needed_bits - 2 + i_high
    # i_low from LSB  = needed_bits - 2 + i_low  (= 0 + 16 的 guard)

    shift = needed_bits - 2 + i_low
    if shift < 0:
        four_over_pi_int <<= (-shift)
        shift = 0

    window = (four_over_pi_int >> shift) & ((1 << W) - 1)
    return window


# ─────────────────────────────────────────────
# 参数归约主函数
# ─────────────────────────────────────────────

class ArgReduceResult:
    """参数归约输出"""
    __slots__ = [
        'exn',      # 异常编码 (0=zero, 1=normal, 2=inf, 3=nan)
        'S_x',      # 输入符号
        'k',        # k mod 8 (3 bits)
        'y_sign',   # y 的符号 (0=正, 1=负)
        'Y_fix',    # |y| 的定点表示 (w_F+g 位, 无符号, 纯小数)
        'E_y',      # |y| 浮点指数 (unbiased, 即 Ex - E0)
        'M_y',      # |y| 浮点尾数 (w_F+1 位定点整数, 含隐含位 1.xxxx)
    ]

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, 0)

    def __repr__(self):
        return (f"ArgReduceResult(exn={self.exn}, S_x={self.S_x}, "
                f"k={self.k}, y_sign={self.y_sign}, "
                f"Y_fix=0x{self.Y_fix:x}, E_y={self.E_y}, M_y=0x{self.M_y:x})")


def argument_reduce(x: float, cfg: HWConfig = DEFAULT_CONFIG) -> ArgReduceResult:
    """
    参数归约：从浮点 x 计算 k 和 y。

    x × 4/π = k + y,  k = round(x × 4/π),  y ∈ [-1/2, 1/2]

    返回 ArgReduceResult，包含 k(mod 8) 和 y 的定点/浮点双格式。
    """
    result = ArgReduceResult()

    # ── 第 0 步：拆解浮点字段 ──
    exn, S_x, E_stored, F = float_to_fields(x, cfg)
    result.exn = exn
    result.S_x = S_x

    # 异常输入提前返回
    if exn != 1:  # 非 normal
        return result

    # ── 第 1 步：计算无偏指数 ──
    # 文档 §4.4.1: E = Ex - E0 + 1 (额外 +1 补偿隐含位对齐)
    E = E_stored - cfg.E0 + 1

    # ── 第 2 步：构造完整尾数 1.Fx ──
    # mantissa_int = (1 << w_F) | F，共 (w_F + 1) 位
    w_F = cfg.w_F
    mantissa = (1 << w_F) | F  # (w_F+1) 位整数

    # ── 第 3 步：提取 4/π 窗口 ──
    C = extract_window(E, cfg)
    W = cfg.window_width

    # ── 第 4 步：定点乘法 ──
    # P = mantissa × C
    # mantissa 是 (w_F+1) 位, C 是 W 位
    # 乘积 P 共 (w_F + 1 + W) 位
    P = mantissa * C

    # ── 第 5 步：从乘积中提取 k 和 y ──
    #
    # 乘积位结构 (从高到低):
    #   [污染区: w_F 位] [k: 3 位] [y 小数: w_F+g+g_K 位] [超精度]
    #
    # 超精度部分宽度 = (w_F+1+W) - w_F - 3 - (w_F+g+g_K)
    #                = (w_F+1) + (2*w_F+g+g_K+3) - w_F - 3 - (w_F+g+g_K)
    #                = w_F + 1
    # 所以超精度部分有 (w_F+1) 位

    y_total_bits = w_F + cfg.g + cfg.g_K  # y 小数位数
    extra_bits = w_F + 1                   # 超精度位数

    # 截掉超精度，保留 k + y 的位
    P_trimmed = P >> extra_bits

    # y 的小数部分 (无符号)
    y_raw = P_trimmed & ((1 << y_total_bits) - 1)

    # k 的 3 位
    k_raw = (P_trimmed >> y_total_bits) & 0x7

    # ── 第 6 步：就近取整 (round to nearest) ──
    # y_raw 是 x*4/π 小数部分的定点表示 (y_total_bits 位)
    # 最高位 (位号 -1) 是 0.5 的位置
    # 如果 y_raw 的最高位 = 1，说明小数 >= 0.5，需要 k+1，y 变为负数

    half = 1 << (y_total_bits - 1)  # 0.5 对应的定点值

    if y_raw >= half:
        # 小数部分 >= 0.5, 向上取整
        k_rounded = (k_raw + 1) & 0x7  # mod 8
        # y = 小数部分 - 1 (负数)
        # |y| = 1 - 小数部分 = (1 << y_total_bits) - y_raw
        y_abs = (1 << y_total_bits) - y_raw
        y_sign = 1  # 负
    else:
        k_rounded = k_raw
        y_abs = y_raw
        y_sign = 0  # 正

    result.k = k_rounded
    result.y_sign = y_sign

    # ── 第 7 步：生成定点 Y (给 cos 通路) ──
    # 从 y_abs (y_total_bits = w_F+g+g_K 位) 截取高 (w_F+g) 位
    y_fix_bits = cfg.y_fixedpoint_bits  # w_F + g
    shift_for_fix = cfg.g_K             # 截掉低 g_K 位
    Y_fix = y_abs >> shift_for_fix

    result.Y_fix = Y_fix

    # ── 第 8 步：生成浮点 (E_y, M_y) (给 sin 通路) ──
    # 对 y_abs 做前导零计数 + 归一化
    if y_abs == 0:
        result.E_y = -(y_total_bits)  # 非常小
        result.M_y = 0
    else:
        # y_abs 是 y_total_bits 位的定点数，值 = y_abs / 2^y_total_bits
        # 找前导零数 (从 MSB 即位号 y_total_bits-1 开始)
        lzc = 0
        for b in range(y_total_bits - 1, -1, -1):
            if (y_abs >> b) & 1:
                break
            lzc += 1

        # 归一化: 左移 lzc+1 位使最高位到 "1.xxx" 格式
        # y = y_abs / 2^y_total_bits
        # y_abs 的最高 '1' 在位号 (y_total_bits - 1 - lzc)
        # 浮点值 = 2^{-(lzc+1)} × (1.xxxx...)
        E_y = -(lzc + 1)

        # 提取 (w_F+1) 位尾数 (含隐含位)
        # 将 y_abs 左移到最高位对齐，然后取高 (w_F+1) 位
        leading_pos = y_total_bits - 1 - lzc  # '1' 所在的位号
        if leading_pos >= w_F:
            M_y = (y_abs >> (leading_pos - w_F)) & ((1 << (w_F + 1)) - 1)
        else:
            M_y = (y_abs << (w_F - leading_pos)) & ((1 << (w_F + 1)) - 1)

        result.E_y = E_y
        result.M_y = M_y

    return result


# ─────────────────────────────────────────────
# 调试/验证用：高精度参考实现
# ─────────────────────────────────────────────

def argument_reduce_reference(x: float):
    """
    用 mpmath 高精度计算参考值，用于验证硬件仿真。
    硬件始终对 |x| 做归约，因此此处也对 |x| 操作。

    Returns:
        k_mod8: k mod 8 (对 |x| 的归约)
        y:      |x|*4/π - k (浮点数, ∈ [-0.5, 0.5])
    """
    mp.prec = 200
    x_mp = mpf(abs(x))
    product = x_mp * 4 / pi
    k = int(nint(product))
    y = float(product - k)
    k_mod8 = k % 8
    return k_mod8, y
