# -*- coding: utf-8 -*-
"""
技术指标计算核心函数
基于EMA算法，使用numba JIT加速实现O(1)时间复杂度的指标计算
"""

import numpy as np
from numba import jit
import math


@jit(nopython=True)
def compute_indicators_batch_multi(timestamps, prices, volumes,
                                 tau_p,  # PEMA时间常数
                                 tau_o,  # 波动率时间常数（独立参数）
                                 sigma_multis,  # 多个波动率缩放因子数组
                                 min_volatility=0.001,  # 波动率下限（默认千分之一）
                                 ticksize=0.01,  # 价格最小单位
                                 time_unit=1000.0,
                                 eps=1e-12,
                                 dt_min=1e-3):
    """
    批量计算技术指标（支持多个sigma_multi参数）

    基于EMA算法计算四大核心指标：
    1. VEMA: 单位时间成交量强度（使用tau_p）
    2. PEMA: 体量加权公允价格（使用tau_p）
    3. VSQEMA: 体量加权价格偏移均线（使用tau_o，独立参数）
    4. 波动率和多组挂单位置（使用tau_o）

    参数：
        timestamps: 时间戳数组（毫秒）
        prices: 价格数组
        volumes: 成交量数组
        tau_p: PEMA时间常数（秒）
        tau_o: 波动率时间常数（秒），独立于tau_p
        sigma_multis: 多个波动率缩放因子数组，如[0.3, 0.5, 1.0, 1.5, 2.0]
        min_volatility: 波动率下限（默认0.001，即千分之一），防止价差过窄
        time_unit: 时间单位转换（默认1000.0，毫秒转秒）
        eps: 数值防护参数
        dt_min: 最小时间间隔（秒）

    返回：
        (vema数组, pema_prior数组, pema数组, vsqema数组, vema_o数组,
         msq_per_trade数组, sigma数组, fair数组,
         多组ask_permit数组字典, 多组bid_permit数组字典)
    """
    n = len(timestamps)
    n_multis = len(sigma_multis)

    # 初始化输出数组
    vema = np.zeros(n, dtype=np.float64)
    pema_prior = np.zeros(n, dtype=np.float64)
    pema = np.zeros(n, dtype=np.float64)
    vsqema = np.zeros(n, dtype=np.float64)
    vema_o = np.zeros(n, dtype=np.float64)
    msq_per_trade = np.zeros(n, dtype=np.float64)
    sigma = np.zeros(n, dtype=np.float64)
    fair = np.zeros(n, dtype=np.float64)

    # 多组挂单位置数组
    ask_permits = np.zeros((n, n_multis), dtype=np.float64)
    bid_permits = np.zeros((n, n_multis), dtype=np.float64)

    # 初始化状态变量
    A_v = 0.0      # VEMA累积量
    A_p = 0.0      # PEMA累积体量
    B_p = 0.0      # PEMA累积加权价格
    A_o = 0.0      # VSQEMA累积体量
    D = 0.0        # VSQEMA累积加权偏移

    # 初始化当前PEMA值
    current_pema = prices[0] if n > 0 else 0.0

    for i in range(n):
        # 时间转换为秒
        t_i = timestamps[i] / time_unit

        # 计算时间差
        if i == 0:
            dt = 0.0
        else:
            t_prev = timestamps[i-1] / time_unit
            dt = max(t_i - t_prev, dt_min)

        # 当前trade数据
        p_i = prices[i]
        v_i = volumes[i]

        # 计算衰减因子（分别计算PEMA和波动率的衰减因子）
        dec_p = math.exp(-dt / tau_p) if dt > 0 else 1.0  # PEMA衰减因子
        dec_o = math.exp(-dt / tau_o) if dt > 0 else 1.0  # 波动率衰减因子

        # 1. 更新VEMA（使用tau_p）
        A_v = A_v * dec_p + v_i / tau_p
        vema[i] = A_v

        # 2. 更新PEMA（记录更新前的值，使用tau_p）
        pema_prior[i] = current_pema

        A_p = A_p * dec_p + v_i / tau_p
        B_p = B_p * dec_p + p_i * v_i / tau_p
        current_pema = B_p / max(A_p, eps)
        pema[i] = current_pema

        # 3. 更新VSQEMA（使用tau_o，独立于PEMA）
        # 计算价格偏移（线性）
        if current_pema > eps:
            e_i = current_pema * math.log(p_i / current_pema)
        else:
            e_i = 0.0

        # 更新累积变量（使用tau_o）
        A_o = A_o * dec_o + v_i / tau_o
        D = D * dec_o + v_i * abs(e_i) / tau_o

        vsqema[i] = D
        vema_o[i] = A_o
        msq_per_trade[i] = D / max(A_o, eps)

        # 4. 计算即时波动率（应用波动率下限约束）
        raw_sigma = msq_per_trade[i]
        # 波动率下限：相对于pema的比例（与固定波动率保持一致的口径）
        sigma_floor = current_pema * min_volatility
        # 同时保证最小值不小于2个tick（避免极端情况）
        min_sigma = max(sigma_floor, ticksize * 2)
        sigma[i] = max(raw_sigma, min_sigma)

        # 5. 计算fair价格（使用最近邻进位）
        fair[i] = round(current_pema / ticksize) * ticksize

        # 6. 计算多组挂单位置（应用不同sigma_multi缩放和ticksize进位）
        for j in range(n_multis):
            sigma_multi = sigma_multis[j]
            # ask_permit向上进位（更保守）
            ask_permits[i, j] = math.ceil((current_pema + sigma[i] * sigma_multi) / ticksize) * ticksize
            # bid_permit向下进位（更保守）
            bid_permits[i, j] = math.floor((current_pema - sigma[i] * sigma_multi) / ticksize) * ticksize

    return (vema, pema_prior, pema, vsqema, vema_o,
            msq_per_trade, sigma, fair, ask_permits, bid_permits)


@jit(nopython=True)
def compute_indicators_batch(timestamps, prices, volumes,
                           tau_p,  # PEMA时间常数
                           tau_o,  # 波动率时间常数（独立参数）
                           sigma_multi=1.0,  # 波动率缩放因子
                           min_volatility=0.001,  # 波动率下限（默认千分之一）
                           ticksize=0.01,  # 价格最小单位
                           time_unit=1000.0,
                           eps=1e-12,
                           dt_min=1e-3):
    """
    批量计算技术指标（numba加速版本）

    基于EMA算法计算四大核心指标：
    1. VEMA: 单位时间成交量强度（使用tau_p）
    2. PEMA: 体量加权公允价格（使用tau_p）
    3. VSQEMA: 体量加权价格偏移均线（使用tau_o，独立参数）
    4. 波动率和挂单位置（使用tau_o）

    参数：
        timestamps: 时间戳数组（毫秒）
        prices: 价格数组
        volumes: 成交量数组
        tau_p: PEMA时间常数（秒）
        tau_o: 波动率时间常数（秒），独立于tau_p
        sigma_multi: 波动率缩放因子（默认1.0）
        min_volatility: 波动率下限（默认0.001，即千分之一），防止价差过窄
        time_unit: 时间单位转换（默认1000.0，毫秒转秒）
        eps: 数值防护参数
        dt_min: 最小时间间隔（秒）

    返回：
        (vema数组, pema_prior数组, pema数组, vsqema数组, vema_o数组,
         msq_per_trade数组, sigma数组, fair数组, ask_permit数组, bid_permit数组)
    """
    n = len(timestamps)

    # 初始化输出数组
    vema = np.zeros(n, dtype=np.float64)
    pema_prior = np.zeros(n, dtype=np.float64)
    pema = np.zeros(n, dtype=np.float64)
    vsqema = np.zeros(n, dtype=np.float64)
    vema_o = np.zeros(n, dtype=np.float64)
    msq_per_trade = np.zeros(n, dtype=np.float64)
    sigma = np.zeros(n, dtype=np.float64)
    fair = np.zeros(n, dtype=np.float64)
    ask_permit = np.zeros(n, dtype=np.float64)
    bid_permit = np.zeros(n, dtype=np.float64)

    # 初始化状态变量
    A_v = 0.0      # VEMA累积量
    A_p = 0.0      # PEMA累积体量
    B_p = 0.0      # PEMA累积加权价格
    A_o = 0.0      # VSQEMA累积体量
    D = 0.0        # VSQEMA累积加权偏移

    # 初始化当前PEMA值
    current_pema = prices[0] if n > 0 else 0.0

    for i in range(n):
        # 时间转换为秒
        t_i = timestamps[i] / time_unit

        # 计算时间差
        if i == 0:
            dt = 0.0
        else:
            t_prev = timestamps[i-1] / time_unit
            dt = max(t_i - t_prev, dt_min)

        # 当前trade数据
        p_i = prices[i]
        v_i = volumes[i]

        # 计算衰减因子（分别计算PEMA和波动率的衰减因子）
        dec_p = math.exp(-dt / tau_p) if dt > 0 else 1.0  # PEMA衰减因子
        dec_o = math.exp(-dt / tau_o) if dt > 0 else 1.0  # 波动率衰减因子

        # 1. 更新VEMA（使用tau_p）
        A_v = A_v * dec_p + v_i / tau_p
        vema[i] = A_v

        # 2. 更新PEMA（记录更新前的值，使用tau_p）
        pema_prior[i] = current_pema

        A_p = A_p * dec_p + v_i / tau_p
        B_p = B_p * dec_p + p_i * v_i / tau_p
        current_pema = B_p / max(A_p, eps)
        pema[i] = current_pema

        # 3. 更新VSQEMA（使用tau_o，独立于PEMA）
        # 计算价格偏移（线性）
        if current_pema > eps:
            e_i = current_pema * math.log(p_i / current_pema)
        else:
            e_i = 0.0

        # 更新累积变量（使用tau_o）
        A_o = A_o * dec_o + v_i / tau_o
        D = D * dec_o + v_i * abs(e_i) / tau_o

        vsqema[i] = D
        vema_o[i] = A_o
        msq_per_trade[i] = D / max(A_o, eps)

        # 4. 计算即时波动率（应用波动率下限约束）
        raw_sigma = msq_per_trade[i]
        # 波动率下限：相对于pema的比例（与固定波动率保持一致的口径）
        sigma_floor = current_pema * min_volatility
        # 同时保证最小值不小于2个tick（避免极端情况）
        min_sigma = max(sigma_floor, ticksize * 2)
        sigma[i] = max(raw_sigma, min_sigma)

        # 5. 计算挂单位置（应用sigma_multi缩放和ticksize进位）
        # fair使用最近邻进位
        fair[i] = round(current_pema / ticksize) * ticksize
        # ask_permit向上进位（更保守）
        ask_permit[i] = math.ceil((current_pema + sigma[i] * sigma_multi) / ticksize) * ticksize
        # bid_permit向下进位（更保守）
        bid_permit[i] = math.floor((current_pema - sigma[i] * sigma_multi) / ticksize) * ticksize

    return (vema, pema_prior, pema, vsqema, vema_o,
            msq_per_trade, sigma, fair, ask_permit, bid_permit)




def parse_time_string(time_str):
    """
    解析时间字符串为秒数

    支持格式：
        - '100s' 或 '100' -> 100秒
        - '5min' 或 '5m' -> 300秒
        - '3h' 或 '3hour' -> 10800秒
        - '1d' 或 '1day' -> 86400秒

    参数：
        time_str: 时间字符串或数值

    返回：
        秒数（浮点数）
    """
    if isinstance(time_str, (int, float)):
        return float(time_str)

    time_str = str(time_str).strip().lower()

    # 提取数字和单位
    import re
    match = re.match(r'^(\d+\.?\d*)\s*([a-z]*)$', time_str)
    if not match:
        raise ValueError(f"无法解析时间字符串: {time_str}")

    value = float(match.group(1))
    unit = match.group(2)

    # 单位转换
    units = {
        '': 1,        # 默认秒
        's': 1,       # 秒
        'sec': 1,
        'second': 1,
        'seconds': 1,
        'm': 60,      # 分钟
        'min': 60,
        'minute': 60,
        'minutes': 60,
        'h': 3600,    # 小时
        'hr': 3600,
        'hour': 3600,
        'hours': 3600,
        'd': 86400,   # 天
        'day': 86400,
        'days': 86400,
    }

    if unit not in units:
        raise ValueError(f"不支持的时间单位: {unit}")

    return value * units[unit]


@jit(nopython=True)
def compute_indicators_fixed_volatility(timestamps, prices, volumes,
                                      tau_p,  # PEMA时间常数
                                      fixed_volatility,  # 固定波动率（如0.001表示千分之一）
                                      ticksize=0.01,  # 价格最小单位
                                      time_unit=1000.0,
                                      eps=1e-12,
                                      dt_min=1e-3):
    """
    计算技术指标（固定波动率版本）

    使用固定的波动率而非动态计算，挂单位置计算方式：
    - ask_permit = pema * (1 + fixed_volatility)
    - bid_permit = pema * (1 - fixed_volatility)

    参数：
        timestamps: 时间戳数组（毫秒）
        prices: 价格数组
        volumes: 成交量数组
        tau_p: PEMA时间常数（秒）
        fixed_volatility: 固定波动率（如0.001表示千分之一，0.002表示千分之二）
        ticksize: 价格最小单位
        time_unit: 时间单位转换（默认1000.0，毫秒转秒）
        eps: 数值防护参数
        dt_min: 最小时间间隔（秒）

    返回：
        (vema数组, pema_prior数组, pema数组, fair数组, ask_permit数组, bid_permit数组,
         固定sigma数组用于兼容性)
    """
    n = len(timestamps)

    # 初始化输出数组
    vema = np.zeros(n, dtype=np.float64)
    pema_prior = np.zeros(n, dtype=np.float64)
    pema = np.zeros(n, dtype=np.float64)
    fair = np.zeros(n, dtype=np.float64)
    ask_permit = np.zeros(n, dtype=np.float64)
    bid_permit = np.zeros(n, dtype=np.float64)
    # 为兼容性返回固定的sigma值数组
    sigma = np.zeros(n, dtype=np.float64)

    # 初始化状态变量
    A_v = 0.0      # VEMA累积量
    A_p = 0.0      # PEMA累积体量
    B_p = 0.0      # PEMA累积加权价格

    # 初始化当前PEMA值
    current_pema = prices[0] if n > 0 else 0.0

    for i in range(n):
        # 时间转换为秒
        t_i = timestamps[i] / time_unit

        # 计算时间差
        if i == 0:
            dt = 0.0
        else:
            t_prev = timestamps[i-1] / time_unit
            dt = max(t_i - t_prev, dt_min)

        # 当前trade数据
        p_i = prices[i]
        v_i = volumes[i]

        # 计算PEMA衰减因子
        dec_p = math.exp(-dt / tau_p) if dt > 0 else 1.0

        # 1. 更新VEMA（使用tau_p）
        A_v = A_v * dec_p + v_i / tau_p
        vema[i] = A_v

        # 2. 更新PEMA（记录更新前的值）
        pema_prior[i] = current_pema

        A_p = A_p * dec_p + v_i / tau_p
        B_p = B_p * dec_p + p_i * v_i / tau_p
        current_pema = B_p / max(A_p, eps)
        pema[i] = current_pema

        # 3. 计算fair价格（使用最近邻进位）
        fair[i] = round(current_pema / ticksize) * ticksize

        # 4. 使用固定波动率计算挂单位置
        # 固定波动率表示的是相对于pema的比例偏移
        # 例如：fixed_volatility=0.001表示千分之一
        # ask_permit向上进位（更保守）
        ask_permit[i] = math.ceil(current_pema * (1 + fixed_volatility) / ticksize) * ticksize
        # bid_permit向下进位（更保守）
        bid_permit[i] = math.floor(current_pema * (1 - fixed_volatility) / ticksize) * ticksize

        # 5. 为兼容性设置固定的sigma值（转换为绝对价格偏移）
        sigma[i] = current_pema * fixed_volatility

    return (vema, pema_prior, pema, fair, ask_permit, bid_permit, sigma)


@jit(nopython=True)
def compute_indicators_fixed_volatility_multi(timestamps, prices, volumes,
                                            tau_p,  # PEMA时间常数
                                            fixed_volatilities,  # 多个固定波动率数组
                                            ticksize=0.01,  # 价格最小单位
                                            time_unit=1000.0,
                                            eps=1e-12,
                                            dt_min=1e-3):
    """
    计算技术指标（多个固定波动率版本）

    支持同时计算多个固定波动率的挂单位置，用于参数扫描

    参数：
        timestamps: 时间戳数组（毫秒）
        prices: 价格数组
        volumes: 成交量数组
        tau_p: PEMA时间常数（秒）
        fixed_volatilities: 固定波动率数组，如[0.001, 0.002, 0.003]
        ticksize: 价格最小单位
        time_unit: 时间单位转换（默认1000.0，毫秒转秒）
        eps: 数值防护参数
        dt_min: 最小时间间隔（秒）

    返回：
        (vema数组, pema_prior数组, pema数组, fair数组,
         多组ask_permit数组, 多组bid_permit数组, 固定sigma数组)
    """
    n = len(timestamps)
    n_vols = len(fixed_volatilities)

    # 初始化输出数组
    vema = np.zeros(n, dtype=np.float64)
    pema_prior = np.zeros(n, dtype=np.float64)
    pema = np.zeros(n, dtype=np.float64)
    fair = np.zeros(n, dtype=np.float64)

    # 多组挂单位置数组
    ask_permits = np.zeros((n, n_vols), dtype=np.float64)
    bid_permits = np.zeros((n, n_vols), dtype=np.float64)
    sigmas = np.zeros((n, n_vols), dtype=np.float64)

    # 初始化状态变量
    A_v = 0.0      # VEMA累积量
    A_p = 0.0      # PEMA累积体量
    B_p = 0.0      # PEMA累积加权价格

    # 初始化当前PEMA值
    current_pema = prices[0] if n > 0 else 0.0

    for i in range(n):
        # 时间转换为秒
        t_i = timestamps[i] / time_unit

        # 计算时间差
        if i == 0:
            dt = 0.0
        else:
            t_prev = timestamps[i-1] / time_unit
            dt = max(t_i - t_prev, dt_min)

        # 当前trade数据
        p_i = prices[i]
        v_i = volumes[i]

        # 计算PEMA衰减因子
        dec_p = math.exp(-dt / tau_p) if dt > 0 else 1.0

        # 1. 更新VEMA（使用tau_p）
        A_v = A_v * dec_p + v_i / tau_p
        vema[i] = A_v

        # 2. 更新PEMA（记录更新前的值）
        pema_prior[i] = current_pema

        A_p = A_p * dec_p + v_i / tau_p
        B_p = B_p * dec_p + p_i * v_i / tau_p
        current_pema = B_p / max(A_p, eps)
        pema[i] = current_pema

        # 3. 计算fair价格（使用最近邻进位）
        fair[i] = round(current_pema / ticksize) * ticksize

        # 4. 计算多组固定波动率的挂单位置
        for j in range(n_vols):
            fixed_vol = fixed_volatilities[j]
            # ask_permit向上进位（更保守）
            ask_permits[i, j] = math.ceil(current_pema * (1 + fixed_vol) / ticksize) * ticksize
            # bid_permit向下进位（更保守）
            bid_permits[i, j] = math.floor(current_pema * (1 - fixed_vol) / ticksize) * ticksize
            # 为兼容性设置sigma值
            sigmas[i, j] = current_pema * fixed_vol

    return (vema, pema_prior, pema, fair, ask_permits, bid_permits, sigmas)


def get_default_parameters():
    """
    获取默认参数配置

    返回：
        默认参数字典
    """
    return {
        'tau_p': 300.0,      # PEMA时间常数，5分钟
        'tau_o': 300.0,      # 波动率时间常数，5分钟（可独立设置）
        'sigma_multi': 1.0,  # 默认不缩放
        'min_volatility': 0.001,  # 波动率下限（千分之一）
        'fixed_volatility': None,  # 固定波动率（None表示使用动态波动率）
        'time_unit': 1000.0, # 毫秒转秒
        'eps': 1e-12,
        'dt_min': 1e-3
    }