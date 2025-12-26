# -*- coding: utf-8 -*-
"""
基于Orderbook的做市策略 - 核心计算逻辑（优化版）

优化方案:
- 方案A: 增量式直方图更新（O(N) → O(1)）
- 方案B: 消除循环缓冲区重排序
- 方案C: CDF预计算优化概率查找（O(N) → O(1)）

预期加速: ~6倍
"""

import numpy as np
from numba import jit, int64, float64


@jit(nopython=True, cache=True)
def compute_cdf_from_histogram(hist, n_bins):
    """
    从直方图计算累计分布函数（从大到小累计）

    参数:
        hist: 直方图概率数组，shape=(max_bins,)
        n_bins: 实际箱体数量

    返回:
        cdf: 累计分布数组，cdf[i] = P(X >= bin_edges[i])

    复杂度: O(n_bins)

    示例:
        hist = [0.3, 0.5, 0.2]  (3个箱体)
        cdf  = [1.0, 0.7, 0.2]  (从大到小累计)

        解释:
        - cdf[0] = 0.3 + 0.5 + 0.2 = 1.0  (所有样本)
        - cdf[1] = 0.5 + 0.2 = 0.7        (箱体1及以上)
        - cdf[2] = 0.2                    (箱体2及以上)
    """
    cdf = np.zeros(n_bins, dtype=np.float64)
    cumsum = 0.0

    # 从最大箱体反向累计
    for i in range(n_bins - 1, -1, -1):
        cumsum += hist[i]
        cdf[i] = cumsum

    return cdf


@jit(nopython=True, cache=True)
def get_fill_probability_fast(cdf, bin_edges, n_bins, bin_width, cumulative_usdt):
    """
    使用预计算的CDF快速查找概率

    参数:
        cdf: 累计分布数组
        bin_edges: 箱体边界
        n_bins: 箱体数量
        bin_width: 箱体宽度
        cumulative_usdt: 目标累计金额

    返回:
        P(流量 >= cumulative_usdt)

    复杂度: O(1)（vs 原来的O(n_bins)）
    """
    if n_bins == 0:
        return 0.0

    # 计算目标金额所在的箱体索引
    min_val = bin_edges[0]

    # 边界情况：低于最小箱体
    if cumulative_usdt < min_val:
        return 1.0  # 所有样本都 >= cumulative_usdt

    # 计算箱体索引
    bin_idx = int((cumulative_usdt - min_val) / bin_width)

    # 边界情况：超出最大箱体
    if bin_idx >= n_bins:
        return 0.0  # 没有样本 >= cumulative_usdt

    # 查表返回概率
    # 注意：我们要找"箱体上界 >= cumulative_usdt"的概率
    # 如果cumulative_usdt落在箱体i内，则从箱体i+1开始累计
    if bin_idx + 1 < n_bins:
        return cdf[bin_idx + 1]
    else:
        return 0.0


@jit(nopython=True, cache=True)
def initialize_histogram_from_ring_buffer(buffer, start_idx, capacity, bin_width, max_bins):
    """
    从环形缓冲区初始化直方图（无需重排序）

    参数:
        buffer: 环形缓冲区
        start_idx: 当前读取起始位置
        capacity: 缓冲区容量
        bin_width: 箱体宽度
        max_bins: 最大箱体数

    返回:
        hist, bin_edges, n_bins, min_val, max_val

    复杂度: O(capacity)（但只在预热阶段调用一次）
    """
    # 1. 找数据范围（使用环形索引）
    min_val = np.inf
    max_val = -np.inf

    for i in range(capacity):
        ring_idx = (start_idx + i) % capacity
        val = buffer[ring_idx]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val

    # 边界情况：所有数据相同
    if max_val == min_val:
        hist = np.zeros(max_bins, dtype=np.float64)
        bin_edges = np.zeros(max_bins + 1, dtype=np.float64)
        hist[0] = 1.0
        bin_edges[0] = min_val
        bin_edges[1] = min_val + bin_width
        return hist, bin_edges, int64(1), min_val, max_val

    # 2. 计算箱体数量
    n_bins = int(np.ceil((max_val - min_val) / bin_width)) + 1
    n_bins = min(n_bins, max_bins)

    # 3. 初始化
    hist = np.zeros(max_bins, dtype=np.float64)
    bin_edges = np.zeros(max_bins + 1, dtype=np.float64)

    # 4. 生成箱体边界
    for i in range(n_bins + 1):
        bin_edges[i] = min_val + i * bin_width

    # 5. 统计（使用环形索引）
    for i in range(capacity):
        ring_idx = (start_idx + i) % capacity
        val = buffer[ring_idx]

        bin_idx = int((val - min_val) / bin_width)
        bin_idx = min(max(bin_idx, 0), n_bins - 1)

        hist[bin_idx] += 1.0

    # 6. 归一化
    for i in range(n_bins):
        hist[i] /= float(capacity)

    return hist, bin_edges, int64(n_bins), min_val, max_val


@jit(nopython=True, cache=True)
def update_histogram_incremental(hist, n_bins, bin_width, min_val,
                                  old_value, new_value, capacity):
    """
    增量更新直方图（方案A核心）

    参数:
        hist: 当前直方图（原地修改）
        n_bins: 箱体数量
        bin_width: 箱体宽度
        min_val: 箱体边界基准
        old_value: 移除的样本值
        new_value: 新增的样本值
        capacity: 总样本数（用于归一化）

    返回:
        new_min_val, new_max_val（用于检测是否需要重建直方图）

    复杂度: O(1)

    注意: 如果new_value超出当前箱体范围，需要外部重建直方图
    """
    norm_factor = 1.0 / float(capacity)

    # 移除旧样本
    old_bin_idx = int((old_value - min_val) / bin_width)
    if 0 <= old_bin_idx < n_bins:
        hist[old_bin_idx] -= norm_factor
        if hist[old_bin_idx] < 0.0:
            hist[old_bin_idx] = 0.0  # 浮点误差保护

    # 添加新样本
    new_bin_idx = int((new_value - min_val) / bin_width)
    if 0 <= new_bin_idx < n_bins:
        hist[new_bin_idx] += norm_factor
    else:
        # 新样本超出范围，需要重建
        return -1.0, -1.0  # 特殊标记

    return min_val, min_val + n_bins * bin_width


@jit(nopython=True, cache=True)
def interpolate_pema(target_time, pema_times, pema_vals):
    """获取PEMA值（只使用过去数据）"""
    n = len(pema_times)

    if n == 0:
        return 0.0

    if target_time < pema_times[0]:
        return pema_vals[0]

    if target_time >= pema_times[n - 1]:
        return pema_vals[n - 1]

    for i in range(n - 1, -1, -1):
        if pema_times[i] <= target_time:
            return pema_vals[i]

    return pema_vals[0]


@jit(nopython=True, cache=True)
def find_optimal_ask_price_fast(ob_prices, ob_amounts, n_levels,
                                 buy_cdf, buy_bins, buy_n_bins, buy_bin_width,
                                 fair_price, initial_equity):
    """
    遍历ask档位，选择预期收益最大的档位（使用CDF加速）

    优化: 使用预计算的CDF，概率查找从O(n_bins) → O(1)
    """
    best_price = np.nan
    best_profit = -np.inf
    best_prob = 0.0
    best_cumulative = 0.0
    best_order_size = 0.0

    cumulative_usdt = 0.0

    for i in range(n_levels):
        price = ob_prices[i]
        amount = ob_amounts[i]

        cumulative_usdt += price * amount

        # 使用CDF快速查找（O(1)）
        prob = get_fill_probability_fast(buy_cdf, buy_bins, buy_n_bins,
                                         buy_bin_width, cumulative_usdt)

        if price > 0:
            order_size = initial_equity / price
        else:
            continue

        if price <= fair_price:
            continue

        expected_profit = prob * (price - fair_price) * order_size

        if expected_profit > best_profit:
            best_profit = expected_profit
            best_price = price
            best_prob = prob
            best_cumulative = cumulative_usdt
            best_order_size = order_size

    return best_price, best_profit, best_prob, best_cumulative, best_order_size


@jit(nopython=True, cache=True)
def find_optimal_bid_price_fast(ob_prices, ob_amounts, n_levels,
                                 sell_cdf, sell_bins, sell_n_bins, sell_bin_width,
                                 fair_price, initial_equity):
    """
    遍历bid档位，选择预期收益最大的档位（使用CDF加速）

    优化: 使用预计算的CDF，概率查找从O(n_bins) → O(1)
    """
    best_price = np.nan
    best_profit = -np.inf
    best_prob = 0.0
    best_cumulative = 0.0
    best_order_size = 0.0

    cumulative_usdt = 0.0

    for i in range(n_levels):
        price = ob_prices[i]
        amount = ob_amounts[i]

        cumulative_usdt += price * amount

        # 使用CDF快速查找（O(1)）
        prob = get_fill_probability_fast(sell_cdf, sell_bins, sell_n_bins,
                                         sell_bin_width, cumulative_usdt)

        if price > 0:
            order_size = initial_equity / price
        else:
            continue

        if price >= fair_price:
            continue

        expected_profit = prob * (fair_price - price) * order_size

        if expected_profit > best_profit:
            best_profit = expected_profit
            best_price = price
            best_prob = prob
            best_cumulative = cumulative_usdt
            best_order_size = order_size

    return best_price, best_profit, best_prob, best_cumulative, best_order_size


@jit(nopython=True, cache=True)
def compute_orderbook_strategy_core_optimized(
    # Trades数据
    trade_times,
    trade_prices,
    trade_qtys,
    trade_is_buyer_maker,
    # Orderbook数据
    ob_timestamps,
    ob_ask_prices,
    ob_ask_amounts,
    ob_bid_prices,
    ob_bid_amounts,
    ob_depths,
    # PEMA数据
    pema_times,
    pema_vals,
    # 参数
    window_seconds,
    bin_width,
    initial_equity,
    snapshot_interval,
    max_bins=1000
):
    """
    主循环：优化版（方案A+B+C）

    优化点:
    1. 方案B: 直接从环形缓冲区计算直方图，无需重排序
    2. 方案A: 增量更新直方图，O(N) → O(1)
    3. 方案C: 预计算CDF，概率查找O(N) → O(1)

    预期加速: ~6倍
    """
    n_snapshots = len(ob_timestamps)
    capacity = int(window_seconds / snapshot_interval)

    # 初始化循环缓冲区
    buy_flow_buffer = np.zeros(capacity, dtype=np.float64)
    sell_flow_buffer = np.zeros(capacity, dtype=np.float64)
    current_index = int64(0)
    filled_count = int64(0)

    # 初始化直方图状态（增量更新用）
    buy_hist = np.zeros(max_bins, dtype=np.float64)
    buy_bins = np.zeros(max_bins + 1, dtype=np.float64)
    buy_n_bins = int64(0)
    buy_min_val = 0.0
    buy_max_val = 0.0
    buy_initialized = False

    sell_hist = np.zeros(max_bins, dtype=np.float64)
    sell_bins = np.zeros(max_bins + 1, dtype=np.float64)
    sell_n_bins = int64(0)
    sell_min_val = 0.0
    sell_max_val = 0.0
    sell_initialized = False

    # 初始化结果数组
    result_timestamps = np.zeros(n_snapshots, dtype=np.int64)
    result_pema = np.zeros(n_snapshots, dtype=np.float64)
    result_ask_permits = np.full(n_snapshots, np.nan, dtype=np.float64)
    result_bid_permits = np.full(n_snapshots, np.nan, dtype=np.float64)
    result_ask_probs = np.zeros(n_snapshots, dtype=np.float64)
    result_bid_probs = np.zeros(n_snapshots, dtype=np.float64)
    result_buy_flow = np.zeros(n_snapshots, dtype=np.float64)
    result_sell_flow = np.zeros(n_snapshots, dtype=np.float64)
    result_ask_profit = np.zeros(n_snapshots, dtype=np.float64)
    result_bid_profit = np.zeros(n_snapshots, dtype=np.float64)
    result_ask_cumulative = np.zeros(n_snapshots, dtype=np.float64)
    result_bid_cumulative = np.zeros(n_snapshots, dtype=np.float64)
    result_ask_order_size = np.zeros(n_snapshots, dtype=np.float64)
    result_bid_order_size = np.zeros(n_snapshots, dtype=np.float64)
    result_ob_best_ask = np.zeros(n_snapshots, dtype=np.float64)
    result_ob_best_bid = np.zeros(n_snapshots, dtype=np.float64)
    result_ob_best_ask_amount = np.zeros(n_snapshots, dtype=np.float64)
    result_ob_best_bid_amount = np.zeros(n_snapshots, dtype=np.float64)

    trade_idx = int64(0)
    n_trades = len(trade_times)

    # 主循环
    for snap_idx in range(n_snapshots):
        cur_ts = ob_timestamps[snap_idx]

        if snap_idx == 0:
            prev_ts = cur_ts - int64(snapshot_interval * 1000)
        else:
            prev_ts = ob_timestamps[snap_idx - 1]

        # 统计流量
        buy_turnover = 0.0
        sell_turnover = 0.0

        while trade_idx < n_trades and trade_times[trade_idx] < prev_ts:
            trade_idx += 1

        temp_idx = trade_idx
        while temp_idx < n_trades and trade_times[temp_idx] < cur_ts:
            turnover = trade_prices[temp_idx] * trade_qtys[temp_idx]

            if not trade_is_buyer_maker[temp_idx]:
                buy_turnover += turnover
            else:
                sell_turnover += turnover

            temp_idx += 1

        # 保存旧值（用于增量更新）
        old_index = current_index
        old_buy_value = buy_flow_buffer[old_index]
        old_sell_value = sell_flow_buffer[old_index]

        # 更新缓冲区
        buy_flow_buffer[current_index] = buy_turnover
        sell_flow_buffer[current_index] = sell_turnover

        current_index = (current_index + 1) % capacity
        filled_count = min(filled_count + 1, capacity)

        # 检查预热
        if filled_count < capacity:
            result_timestamps[snap_idx] = cur_ts
            result_pema[snap_idx] = interpolate_pema(cur_ts, pema_times, pema_vals)
            continue

        # === 方案A+B: 增量更新直方图 ===

        # 首次预热完成，初始化直方图
        if not buy_initialized:
            (buy_hist, buy_bins, buy_n_bins,
             buy_min_val, buy_max_val) = initialize_histogram_from_ring_buffer(
                buy_flow_buffer, current_index, capacity, bin_width, max_bins
            )
            buy_initialized = True
        else:
            # 增量更新
            new_min, new_max = update_histogram_incremental(
                buy_hist, buy_n_bins, bin_width, buy_min_val,
                old_buy_value, buy_turnover, capacity
            )

            # 检查是否需要重建（新样本超出范围）
            if new_min < 0.0:  # 特殊标记
                (buy_hist, buy_bins, buy_n_bins,
                 buy_min_val, buy_max_val) = initialize_histogram_from_ring_buffer(
                    buy_flow_buffer, current_index, capacity, bin_width, max_bins
                )

        if not sell_initialized:
            (sell_hist, sell_bins, sell_n_bins,
             sell_min_val, sell_max_val) = initialize_histogram_from_ring_buffer(
                sell_flow_buffer, current_index, capacity, bin_width, max_bins
            )
            sell_initialized = True
        else:
            new_min, new_max = update_histogram_incremental(
                sell_hist, sell_n_bins, bin_width, sell_min_val,
                old_sell_value, sell_turnover, capacity
            )

            if new_min < 0.0:
                (sell_hist, sell_bins, sell_n_bins,
                 sell_min_val, sell_max_val) = initialize_histogram_from_ring_buffer(
                    sell_flow_buffer, current_index, capacity, bin_width, max_bins
                )

        # === 方案C: 预计算CDF ===
        buy_cdf = compute_cdf_from_histogram(buy_hist, buy_n_bins)
        sell_cdf = compute_cdf_from_histogram(sell_hist, sell_n_bins)

        # 获取PEMA
        fair_price = interpolate_pema(cur_ts, pema_times, pema_vals)

        # 提取orderbook
        ask_prices = ob_ask_prices[snap_idx, :]
        ask_amounts = ob_ask_amounts[snap_idx, :]
        bid_prices = ob_bid_prices[snap_idx, :]
        bid_amounts = ob_bid_amounts[snap_idx, :]
        n_levels = ob_depths[snap_idx]

        # 计算最优价格（使用CDF加速）
        best_ask, ask_profit, ask_prob, ask_cumulative, ask_order_size = find_optimal_ask_price_fast(
            ask_prices, ask_amounts, n_levels,
            buy_cdf, buy_bins, buy_n_bins, bin_width,
            fair_price, initial_equity
        )

        best_bid, bid_profit, bid_prob, bid_cumulative, bid_order_size = find_optimal_bid_price_fast(
            bid_prices, bid_amounts, n_levels,
            sell_cdf, sell_bins, sell_n_bins, bin_width,
            fair_price, initial_equity
        )

        # 提取1档信息
        ob_best_ask = ask_prices[0] if n_levels > 0 else np.nan
        ob_best_bid = bid_prices[0] if n_levels > 0 else np.nan
        ob_best_ask_amount = ask_amounts[0] if n_levels > 0 else 0.0
        ob_best_bid_amount = bid_amounts[0] if n_levels > 0 else 0.0

        # 保存结果
        result_timestamps[snap_idx] = cur_ts
        result_pema[snap_idx] = fair_price
        result_ask_permits[snap_idx] = best_ask
        result_bid_permits[snap_idx] = best_bid
        result_ask_probs[snap_idx] = ask_prob
        result_bid_probs[snap_idx] = bid_prob
        result_buy_flow[snap_idx] = buy_turnover
        result_sell_flow[snap_idx] = sell_turnover
        result_ask_profit[snap_idx] = ask_profit
        result_bid_profit[snap_idx] = bid_profit
        result_ask_cumulative[snap_idx] = ask_cumulative
        result_bid_cumulative[snap_idx] = bid_cumulative
        result_ask_order_size[snap_idx] = ask_order_size
        result_bid_order_size[snap_idx] = bid_order_size
        result_ob_best_ask[snap_idx] = ob_best_ask
        result_ob_best_bid[snap_idx] = ob_best_bid
        result_ob_best_ask_amount[snap_idx] = ob_best_ask_amount
        result_ob_best_bid_amount[snap_idx] = ob_best_bid_amount

    return (result_timestamps, result_pema,
            result_ask_permits, result_bid_permits,
            result_ask_probs, result_bid_probs,
            result_buy_flow, result_sell_flow,
            result_ask_profit, result_bid_profit,
            result_ask_cumulative, result_bid_cumulative,
            result_ask_order_size, result_bid_order_size,
            result_ob_best_ask, result_ob_best_bid,
            result_ob_best_ask_amount, result_ob_best_bid_amount)
