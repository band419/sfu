import numpy as np
from scipy.interpolate import CubicSpline

# 自适应插值点生成函数
def adaptive_points(func, x_range, num_points):
    """
    根据函数一阶导数的绝对值等间隔选择插值点
    :param func: 目标函数
    :param x_range: 插值范围 (start, end)
    :param num_points: 插值点数量
    :return: 自适应插值点数组
    """
    # 在范围内生成密集的点
    dense_x = np.linspace(x_range[0], x_range[1], 1000)
    dense_y = func(dense_x)
    
    # 计算一阶导数的绝对值
    derivative = np.gradient(dense_y, dense_x)
    abs_derivative = np.abs(derivative)
    
    # 累积分布函数 (CDF) 用于等间隔选择点
    cdf = np.cumsum(abs_derivative)
    cdf = cdf / cdf[-1]  # 归一化到 [0, 1]
    
    # 在 CDF 上等间隔选择点
    target_cdf_values = np.linspace(0, 1, num_points)
    adaptive_x = np.interp(target_cdf_values, cdf, dense_x)
    
    return adaptive_x

# 修改后的 create_interpolator 函数，支持自适应插值点
def create_interpolator(func, x_range, num_points, adaptive=False):
    """
    创建三次样条插值器
    :param func: 要插值的函数
    :param x_range: x的范围 (start, end)
    :param num_points: 分段点的数量
    :param adaptive: 是否使用自适应插值点
    :return: 插值器对象
    """
    if adaptive:
        x_points = adaptive_points(func, x_range, num_points)
    else:
        x_points = np.linspace(x_range[0], x_range[1], num_points)
    y_points = func(x_points)
    return CubicSpline(x_points, y_points)

# 计算理论相对误差的最大值
def calculate_theoretical_relative_error(func, func_fourth_derivative, x_range, num_points):
    """
    计算三次样条插值的理论相对误差最大值
    :param func: 目标函数
    :param func_fourth_derivative: 目标函数的四阶导数
    :param x_range: 插值范围 (start, end)
    :param num_points: 插值点数量
    :return: 理论相对误差的最大值
    """
    # 计算最大间隔 h
    h = (x_range[1] - x_range[0]) / (num_points - 1)
    
    # 计算四阶导数的最大值 M
    x_dense = np.linspace(x_range[0], x_range[1], 1000)  # 用密集点估计最大值
    M = np.max(np.abs(func_fourth_derivative(x_dense)))
    
    # 理论误差公式
    max_absolute_error = (M / 384) * h**4
    
    # 计算目标函数的最小值（避免除以零）
    func_values = np.abs(func(x_dense))
    min_func_value = np.min(func_values)
    
    # 返回理论相对误差最大值
    if min_func_value == 0:
        raise ValueError("目标函数在范围内的值为零，无法计算相对误差。")
    return max_absolute_error / min_func_value

# 示例：计算理论相对误差最大值
if __name__ == "__main__":
    # 定义插值范围和分段点数量
    x_range = (1.18e-38, 3.40e38)  # 避免 x = 0，因为倒数函数在 x = 0 处无定义
    num_points = 1000000000000000000

    # 定义目标函数的四阶导数
    sin_fourth_derivative = lambda x: np.sin(x)  # sin(x) 的四阶导数是 sin(x)
    cos_fourth_derivative = lambda x: np.cos(x)  # cos(x) 的四阶导数是 cos(x)
    exp_fourth_derivative = lambda x: np.exp(x)  # exp(x) 的四阶导数是 exp(x)
    reciprocal_fourth_derivative = lambda x: 24 / x**5  # 1/x 的四阶导数是 24/x^5

    # 计算理论相对误差最大值
    sin_relative_error = calculate_theoretical_relative_error(np.sin, sin_fourth_derivative, x_range, num_points)
    cos_relative_error = calculate_theoretical_relative_error(np.cos, cos_fourth_derivative, x_range, num_points)
    exp_relative_error = calculate_theoretical_relative_error(np.exp, exp_fourth_derivative, (0, 5), num_points)
    reciprocal_relative_error = calculate_theoretical_relative_error(lambda x: 1 / x, reciprocal_fourth_derivative, x_range, num_points)

    # 打印理论相对误差最大值
    print(f"sin(x) 理论相对误差最大值: {sin_relative_error}")
    print(f"cos(x) 理论相对误差最大值: {cos_relative_error}")
    print(f"exp(x) 理论相对误差最大值: {exp_relative_error}")
    print(f"1/x 理论相对误差最大值: {reciprocal_relative_error}")