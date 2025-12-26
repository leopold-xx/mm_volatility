# -*- coding: utf-8 -*-
"""
基于Orderbook的做市策略 - 核心计算逻辑（numba加速）

实现概率驱动的最优档位选择算法:
1. 统计历史成交流量分布
2. 计算每个档位的成交概率
3. 选择预期收益最大的档位

所有核心函数使用numba JIT编译，实现O(1)时间和空间复杂度
"""

import numpy as np
from numba import jit, int64, float64


@jit(nopython=True, cache=True)
def compute_histogram_manual(data, bin_width, max_bins=1000):
    """
    手动计算直方图（numba兼容）

    参数:
        data: 1D数组，流量样本（USDT金额），如买方成交金额的180个点
        bin_width: 箱体宽度（USDT）
        max_bins: 最大箱体数量（预分配，防止内存溢出）

    返回:
        hist: 概率数组，shape=(max_bins,)，hist[i]表示第i个箱体的概率
        bin_edges: 箱体边界，shape=(max_bins+1,)，bin_edges[i]表示第i个箱体的左边界
        n_bins: 实际箱体数量（有效数据在hist[:n_bins]）

    算法:
        1. 计算数据范围[min_val, max_val]
        2. 计算箱体数量：n_bins = ceil((max_val - min_val) / bin_width) + 1
        3. 生成箱体边界：bin_edges[i] = min_val + i * bin_width
        4. 统计每个数据点属于哪个箱体，累计计数
        5. 归一化为概率：hist[i] /= len(data)

    示例:
        输入: data = [1200, 980, 1500, 800, 2100], bin_width = 1000
        输出: min_val = 800, max_val = 2100
              n_bins = 3 (箱体: [800,1800), [1800,2800), [2800,3800))
              hist = [0.4, 0.4, 0.2, 0, ...] (归一化后)
              bin_edges = [800, 1800, 2800, 3800, 0, ...]
    """
    n = len(data)
    if n == 0:
        # 空数据：返回零数组
        return (np.zeros(max_bins, dtype=np.float64),
                np.zeros(max_bins + 1, dtype=np.float64),
                int64(0))

    # 1. 计算数据范围
    min_val = np.min(data)
    max_val = np.max(data)

    # 边界情况：所有数据相同
    if max_val == min_val:
        hist = np.zeros(max_bins, dtype=np.float64)
        bin_edges = np.zeros(max_bins + 1, dtype=np.float64)
        hist[0] = 1.0  # 所有概率在第一个箱体
        bin_edges[0] = min_val
        bin_edges[1] = min_val + bin_width
        return hist, bin_edges, int64(1)

    # 2. 计算箱体数量（不留空箱）
    n_bins = int(np.ceil((max_val - min_val) / bin_width)) + 1
    n_bins = min(n_bins, max_bins)  # 不超过预分配上限

    # 3. 初始化
    hist = np.zeros(max_bins, dtype=np.float64)
    bin_edges = np.zeros(max_bins + 1, dtype=np.float64)

    # 4. 生成箱体边界
    for i in range(n_bins + 1):
        bin_edges[i] = min_val + i * bin_width

    # 5. 统计每个数据点属于哪个箱体
    for i in range(n):
        val = data[i]

        # 计算箱体索引
        bin_idx = int(np.floor((val - min_val) / bin_width))

        # 防止越界（处理浮点误差）
        if bin_idx < 0:
            bin_idx = 0
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1

        # 计数
        hist[bin_idx] += 1.0

    # 6. 归一化为概率
    for i in range(n_bins):
        hist[i] /= float(n)

    return hist, bin_edges, int64(n_bins)


@jit(nopython=True, cache=True)
def get_fill_probability(hist, bin_edges, n_bins, cumulative_usdt):
    """
    计算流量 >= cumulative_usdt 的概率

    参数:
        hist: 直方图概率数组
        bin_edges: 箱体边界数组
        n_bins: 实际箱体数量
        cumulative_usdt: 目标累计金额（USDT）

    返回:
        概率值 [0, 1]

    逻辑:
        累加所有"箱体上界 >= cumulative_usdt"的箱体的概率
        表示历史上流量至少达到cumulative_usdt的频率

    示例:
        箱体1: [800, 1800)   → 概率 30%
        箱体2: [1800, 2800)  → 概率 55%
        箱体3: [2800, 3800)  → 概率 15%

        如果 cumulative_usdt = 2000:
            箱体1上界 1800 < 2000   ✗  → 不累加
            箱体2上界 2800 >= 2000  ✓  → 累加 55%
            箱体3上界 3800 >= 2000  ✓  → 累加 15%
            总概率 = 55% + 15% = 70%
    """
    prob = 0.0

    for i in range(n_bins):
        # 箱体上界
        upper_bound = bin_edges[i + 1]

        # 如果箱体上界 >= 目标金额，累加概率
        if upper_bound >= cumulative_usdt:
            prob += hist[i]

    # 防止浮点误差导致概率>1
    if prob > 1.0:
        prob = 1.0

    return prob


@jit(nopython=True, cache=True)
def interpolate_pema(target_time, pema_times, pema_vals):
    """
    获取target_time时刻的PEMA值（只使用过去的数据，避免未来数据泄露）

    参数:
        target_time: 目标时间戳（毫秒）
        pema_times: PEMA时间戳数组（毫秒，升序）
        pema_vals: PEMA值数组

    返回:
        PEMA值（使用最近的过去值）

    逻辑:
        - 找到 <= target_time 的最后一个时间点
        - 返回该时间点的PEMA值
        - 如果target_time < pema_times[0]，返回第一个值（边界情况）

    示例:
        pema_times = [1000, 1500, 2000, 2500]
        pema_vals  = [0.60, 0.61, 0.62, 0.63]

        target_time = 1800 → 返回 0.61 (使用1500时刻的值)
        target_time = 2000 → 返回 0.62 (精确匹配)
        target_time = 2300 → 返回 0.62 (使用2000时刻的值)
    """
    n = len(pema_times)

    if n == 0:
        return 0.0

    # 边界情况：目标时间在第一个数据点之前
    if target_time < pema_times[0]:
        return pema_vals[0]

    # 边界情况：目标时间在最后一个数据点之后
    if target_time >= pema_times[n - 1]:
        return pema_vals[n - 1]

    # 向后查找：找到 <= target_time 的最后一个时间点
    # 从后往前遍历更高效（假设target_time通常接近末尾）
    for i in range(n - 1, -1, -1):
        if pema_times[i] <= target_time:
            return pema_vals[i]

    # 理论上不会到达这里（已被边界情况覆盖）
    return pema_vals[0]


@jit(nopython=True, cache=True)
def find_optimal_ask_price(ob_prices, ob_amounts, n_levels,
                          buy_hist, buy_bins, buy_n_bins,
                          fair_price, initial_equity):
    """
    遍历ask档位，选择预期收益最大的档位

    参数:
        ob_prices: ask侧价格数组（升序），shape=(max_depth,)
        ob_amounts: ask侧数量数组，shape=(max_depth,)
        n_levels: 实际档位数
        buy_hist: 买方流量直方图（概率）
        buy_bins: 买方流量箱体边界
        buy_n_bins: 买方流量箱体数量
        fair_price: 公允价格（PEMA）
        initial_equity: 单次建仓金额（USDT）

    返回:
        (best_price, best_profit, best_prob, best_cumulative, best_order_size)
        - best_price: 最优ask价格（如果无合适档位则为NaN）
        - best_profit: 最大预期收益（USDT）
        - best_prob: 对应的成交概率
        - best_cumulative: 到达最优档位的累计挂单金额（USDT）
        - best_order_size: 最优档位的挂单量（币）

    算法:
        对每个ask档位（从最优价开始）:
            1. 累计挂单金额 += price * amount
            2. 计算成交概率 = P(买方流量 >= 累计金额)
            3. 计算挂单量 = initial_equity / price
            4. 计算预期收益 = prob * (price - fair_price) * quantity
            5. 约束：price > fair_price
            6. 更新最优解

    注意:
        - ask侧用买方流量（is_buyer_maker=False）
        - 预期收益 = prob * 价差 * 挂单量
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

        # 累计挂单金额
        cumulative_usdt += price * amount

        # 计算成交概率
        prob = get_fill_probability(buy_hist, buy_bins, buy_n_bins, cumulative_usdt)

        # 计算挂单量（空仓建仓）
        if price > 0:
            order_size = initial_equity / price
        else:
            continue  # 价格为0，跳过

        # 约束：价格必须 > 公允价格
        if price <= fair_price:
            continue

        # 计算预期收益
        expected_profit = prob * (price - fair_price) * order_size

        # 更新最优解
        if expected_profit > best_profit:
            best_profit = expected_profit
            best_price = price
            best_prob = prob
            best_cumulative = cumulative_usdt
            best_order_size = order_size

    return best_price, best_profit, best_prob, best_cumulative, best_order_size


@jit(nopython=True, cache=True)
def find_optimal_bid_price(ob_prices, ob_amounts, n_levels,
                          sell_hist, sell_bins, sell_n_bins,
                          fair_price, initial_equity):
    """
    遍历bid档位，选择预期收益最大的档位

    参数:
        ob_prices: bid侧价格数组（降序），shape=(max_depth,)
        ob_amounts: bid侧数量数组，shape=(max_depth,)
        n_levels: 实际档位数
        sell_hist: 卖方流量直方图（概率）
        sell_bins: 卖方流量箱体边界
        sell_n_bins: 卖方流量箱体数量
        fair_price: 公允价格（PEMA）
        initial_equity: 单次建仓金额（USDT）

    返回:
        (best_price, best_profit, best_prob, best_cumulative, best_order_size)
        - best_price: 最优bid价格（如果无合适档位则为NaN）
        - best_profit: 最大预期收益（USDT）
        - best_prob: 对应的成交概率
        - best_cumulative: 到达最优档位的累计挂单金额（USDT）
        - best_order_size: 最优档位的挂单量（币）

    算法:
        对每个bid档位（从最优价开始）:
            1. 累计挂单金额 += price * amount
            2. 计算成交概率 = P(卖方流量 >= 累计金额)
            3. 计算挂单量 = initial_equity / price
            4. 计算预期收益 = prob * (fair_price - price) * quantity
            5. 约束：price < fair_price
            6. 更新最优解

    注意:
        - bid侧用卖方流量（is_buyer_maker=True）
        - 预期收益 = prob * (公允价 - 挂单价) * 挂单量
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

        # 累计挂单金额
        cumulative_usdt += price * amount

        # 计算成交概率
        prob = get_fill_probability(sell_hist, sell_bins, sell_n_bins, cumulative_usdt)

        # 计算挂单量（空仓建仓）
        if price > 0:
            order_size = initial_equity / price
        else:
            continue  # 价格为0，跳过

        # 约束：价格必须 < 公允价格
        if price >= fair_price:
            continue

        # 计算预期收益
        expected_profit = prob * (fair_price - price) * order_size

        # 更新最优解
        if expected_profit > best_profit:
            best_profit = expected_profit
            best_price = price
            best_prob = prob
            best_cumulative = cumulative_usdt
            best_order_size = order_size

    return best_price, best_profit, best_prob, best_cumulative, best_order_size


@jit(nopython=True, cache=True)
def compute_orderbook_strategy_core(
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
    snapshot_interval
):
    """
    主循环：按10秒截面遍历，计算最优挂单价格

    参数:
        trade_times: 逐笔成交时间戳数组（毫秒）
        trade_prices: 逐笔成交价格数组
        trade_qtys: 逐笔成交数量数组
        trade_is_buyer_maker: 逐笔成交买方是否为maker（True=卖方主动）

        ob_timestamps: orderbook截面时间戳数组（毫秒）
        ob_ask_prices: ask侧价格2D数组，shape=(n_snapshots, max_depth)
        ob_ask_amounts: ask侧数量2D数组
        ob_bid_prices: bid侧价格2D数组
        ob_bid_amounts: bid侧数量2D数组
        ob_depths: 每个截面的实际档位数，shape=(n_snapshots,)

        pema_times: PEMA时间戳数组（毫秒）
        pema_vals: PEMA值数组

        window_seconds: 流量统计窗口（秒），如1800表示30分钟
        bin_width: 直方图箱体宽度（USDT）
        initial_equity: 单次建仓金额（USDT）
        snapshot_interval: orderbook采样间隔（秒），固定为10

    返回:
        (result_timestamps, result_pema, result_ask_permits, result_bid_permits,
         result_ask_probs, result_bid_probs,
         result_buy_flow, result_sell_flow,
         result_ask_profit, result_bid_profit,
         result_ask_cumulative, result_bid_cumulative,
         result_ask_order_size, result_bid_order_size,
         result_ob_best_ask, result_ob_best_bid,
         result_ob_best_ask_amount, result_ob_best_bid_amount)

        所有返回数组shape=(n_snapshots,)
        - result_timestamps: 截面时间戳（毫秒）
        - result_pema: 公允价格
        - result_ask_permits: 最优ask价格（可能为NaN）
        - result_bid_permits: 最优bid价格（可能为NaN）
        - result_ask_probs: ask成交概率
        - result_bid_probs: bid成交概率
        - result_buy_flow: 买方流量（USDT/截面）
        - result_sell_flow: 卖方流量（USDT/截面）
        - result_ask_profit: ask预期收益（USDT）
        - result_bid_profit: bid预期收益（USDT）
        - result_ask_cumulative: ask累计挂单金额（USDT）
        - result_bid_cumulative: bid累计挂单金额（USDT）
        - result_ask_order_size: ask挂单量（币）
        - result_bid_order_size: bid挂单量（币）
        - result_ob_best_ask: orderbook最优ask价格
        - result_ob_best_bid: orderbook最优bid价格
        - result_ob_best_ask_amount: orderbook最优ask数量
        - result_ob_best_bid_amount: orderbook最优bid数量

    算法流程:
        1. 初始化循环缓冲区（容量=window_seconds/snapshot_interval）
        2. 遍历每个orderbook截面:
            a. 统计[prev_ts, cur_ts)区间的买卖流量
            b. 更新循环缓冲区
            c. 检查预热状态（是否累积足够数据）
            d. 提取有效数据并计算直方图（买卖分开）
            e. 插值获取PEMA
            f. 遍历档位计算最优ask价格
            g. 遍历档位计算最优bid价格
            h. 保存结果
        3. 返回结果数组
    """
    n_snapshots = len(ob_timestamps)
    max_depth = ob_ask_prices.shape[1]

    # 计算循环缓冲区容量
    capacity = int(window_seconds / snapshot_interval)  # 1800 / 10 = 180

    # 初始化循环缓冲区
    buy_flow_buffer = np.zeros(capacity, dtype=np.float64)   # 买方成交金额
    sell_flow_buffer = np.zeros(capacity, dtype=np.float64)  # 卖方成交金额
    current_index = int64(0)  # 当前写入位置
    filled_count = int64(0)   # 已填充数量

    # 初始化结果数组
    result_timestamps = np.zeros(n_snapshots, dtype=np.int64)
    result_pema = np.zeros(n_snapshots, dtype=np.float64)
    result_ask_permits = np.full(n_snapshots, np.nan, dtype=np.float64)
    result_bid_permits = np.full(n_snapshots, np.nan, dtype=np.float64)
    result_ask_probs = np.zeros(n_snapshots, dtype=np.float64)
    result_bid_probs = np.zeros(n_snapshots, dtype=np.float64)

    # 新增结果数组
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

    # 时间戳索引（用于快速查找trades）
    trade_idx = int64(0)
    n_trades = len(trade_times)

    # 遍历每个orderbook截面
    for snap_idx in range(n_snapshots):
        cur_ts = ob_timestamps[snap_idx]

        # 确定时间范围[prev_ts, cur_ts)
        if snap_idx == 0:
            prev_ts = cur_ts - int64(snapshot_interval * 1000)  # 向前推10秒（毫秒）
        else:
            prev_ts = ob_timestamps[snap_idx - 1]

        # 统计[prev_ts, cur_ts)区间的买卖流量
        buy_turnover = 0.0
        sell_turnover = 0.0

        # 跳过早于prev_ts的trades
        while trade_idx < n_trades and trade_times[trade_idx] < prev_ts:
            trade_idx += 1

        # 收集[prev_ts, cur_ts)区间的trades
        temp_idx = trade_idx
        while temp_idx < n_trades and trade_times[temp_idx] < cur_ts:
            turnover = trade_prices[temp_idx] * trade_qtys[temp_idx]

            if not trade_is_buyer_maker[temp_idx]:
                # 买方主动（卖方是maker）
                buy_turnover += turnover
            else:
                # 卖方主动（买方是maker）
                sell_turnover += turnover

            temp_idx += 1

        # 更新循环缓冲区
        buy_flow_buffer[current_index] = buy_turnover
        sell_flow_buffer[current_index] = sell_turnover

        current_index = (current_index + 1) % capacity
        filled_count = min(filled_count + 1, capacity)

        # 检查预热状态
        if filled_count < capacity:
            # 未累积足够数据，跳过（不挂单）
            result_timestamps[snap_idx] = cur_ts
            result_pema[snap_idx] = interpolate_pema(cur_ts, pema_times, pema_vals)
            continue

        # 提取有效数据（重排序循环队列）
        buy_data = np.concatenate((
            buy_flow_buffer[current_index:],
            buy_flow_buffer[:current_index]
        ))
        sell_data = np.concatenate((
            sell_flow_buffer[current_index:],
            sell_flow_buffer[:current_index]
        ))

        # 计算直方图（买卖分开）
        buy_hist, buy_bins, buy_n_bins = compute_histogram_manual(buy_data, bin_width)
        sell_hist, sell_bins, sell_n_bins = compute_histogram_manual(sell_data, bin_width)

        # 获取公允价格（插值PEMA）
        fair_price = interpolate_pema(cur_ts, pema_times, pema_vals)

        # 提取orderbook数据
        ask_prices = ob_ask_prices[snap_idx, :]
        ask_amounts = ob_ask_amounts[snap_idx, :]
        bid_prices = ob_bid_prices[snap_idx, :]
        bid_amounts = ob_bid_amounts[snap_idx, :]
        n_levels = ob_depths[snap_idx]

        # 计算最优ask价格
        best_ask, ask_profit, ask_prob, ask_cumulative, ask_order_size = find_optimal_ask_price(
            ask_prices, ask_amounts, n_levels,
            buy_hist, buy_bins, buy_n_bins,
            fair_price, initial_equity
        )

        # 计算最优bid价格
        best_bid, bid_profit, bid_prob, bid_cumulative, bid_order_size = find_optimal_bid_price(
            bid_prices, bid_amounts, n_levels,
            sell_hist, sell_bins, sell_n_bins,
            fair_price, initial_equity
        )

        # 提取orderbook 1档信息
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

        # 保存新增字段
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
