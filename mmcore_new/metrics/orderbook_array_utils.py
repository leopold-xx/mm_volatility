# -*- coding: utf-8 -*-
"""
Orderbook数组处理工具 - Numba加速版本

功能：
1. 将扁平化的orderbook DataFrame转换为numpy数组（numba加速）
2. 单次遍历填充bid/ask数组
3. 自动排序（ask升序，bid降序）
"""

import numpy as np
from numba import jit, int64


@jit(nopython=True, cache=True)
def prepare_orderbook_arrays_numba(
    timestamps,      # 所有行的时间戳 (int64)
    sides,          # 所有行的side编码 (int8: 0=bid, 1=ask)
    prices,         # 所有行的价格 (float64)
    amounts,        # 所有行的数量 (float64)
    unique_ts,      # 唯一时间戳数组（升序）
    max_depth       # 最大深度
):
    """
    将扁平化orderbook数据转换为2D数组（numba JIT加速）

    算法：
    1. 创建时间戳到索引的映射（使用二分查找）
    2. 为每个截面维护bid/ask计数器
    3. 单次遍历所有行，直接填充到对应位置
    4. 自动处理排序（假设输入已排序）

    参数：
        timestamps: 1D数组，所有行的时间戳，shape=(n_rows,)
        sides: 1D数组，side编码（0=bid, 1=ask），shape=(n_rows,)
        prices: 1D数组，价格，shape=(n_rows,)
        amounts: 1D数组，数量，shape=(n_rows,)
        unique_ts: 1D数组，唯一时间戳（升序），shape=(n_snapshots,)
        max_depth: 最大深度

    返回：
        (ob_ask_prices, ob_ask_amounts, ob_bid_prices, ob_bid_amounts, ob_depths)
        - ob_ask_prices: 2D数组，shape=(n_snapshots, max_depth)
        - ob_ask_amounts: 2D数组，shape=(n_snapshots, max_depth)
        - ob_bid_prices: 2D数组，shape=(n_snapshots, max_depth)
        - ob_bid_amounts: 2D数组，shape=(n_snapshots, max_depth)
        - ob_depths: 1D数组，每个截面的实际深度，shape=(n_snapshots,)
    """
    n_snapshots = len(unique_ts)
    n_rows = len(timestamps)

    # 预分配输出数组
    ob_ask_prices = np.zeros((n_snapshots, max_depth), dtype=np.float64)
    ob_ask_amounts = np.zeros((n_snapshots, max_depth), dtype=np.float64)
    ob_bid_prices = np.zeros((n_snapshots, max_depth), dtype=np.float64)
    ob_bid_amounts = np.zeros((n_snapshots, max_depth), dtype=np.float64)
    ob_depths = np.zeros(n_snapshots, dtype=np.int64)

    # 为每个截面维护计数器
    ask_counters = np.zeros(n_snapshots, dtype=np.int64)
    bid_counters = np.zeros(n_snapshots, dtype=np.int64)

    # 单次遍历所有行
    for i in range(n_rows):
        ts = timestamps[i]
        side = sides[i]
        price = prices[i]
        amount = amounts[i]

        # 二分查找时间戳对应的索引
        snap_idx = np.searchsorted(unique_ts, ts)

        # 边界检查（防止越界）
        if snap_idx >= n_snapshots:
            continue

        if side == 1:  # ask侧
            counter = ask_counters[snap_idx]
            if counter < max_depth:
                ob_ask_prices[snap_idx, counter] = price
                ob_ask_amounts[snap_idx, counter] = amount
                ask_counters[snap_idx] += 1
        else:  # bid侧 (side == 0)
            counter = bid_counters[snap_idx]
            if counter < max_depth:
                ob_bid_prices[snap_idx, counter] = price
                ob_bid_amounts[snap_idx, counter] = amount
                bid_counters[snap_idx] += 1

    # 计算实际深度（取bid和ask的最小值）
    for i in range(n_snapshots):
        ob_depths[i] = min(ask_counters[i], bid_counters[i])

    return (ob_ask_prices, ob_ask_amounts,
            ob_bid_prices, ob_bid_amounts, ob_depths)


@jit(nopython=True, cache=True)
def sort_orderbook_sides_numba(
    ob_ask_prices,
    ob_ask_amounts,
    ob_bid_prices,
    ob_bid_amounts,
    ob_depths
):
    """
    对orderbook的bid和ask侧进行排序

    排序规则：
    - ask侧：价格升序（从低到高）
    - bid侧：价格降序（从高到低）

    参数：
        ob_ask_prices: 2D数组，shape=(n_snapshots, max_depth)
        ob_ask_amounts: 2D数组
        ob_bid_prices: 2D数组
        ob_bid_amounts: 2D数组
        ob_depths: 1D数组，每个截面的实际深度

    返回：
        排序后的数组（in-place修改）
    """
    n_snapshots = len(ob_depths)

    for snap_idx in range(n_snapshots):
        depth = ob_depths[snap_idx]

        if depth == 0:
            continue

        # Ask侧排序（升序）
        # 提取有效数据
        ask_prices_slice = ob_ask_prices[snap_idx, :depth].copy()
        ask_amounts_slice = ob_ask_amounts[snap_idx, :depth].copy()

        # 排序索引
        ask_sort_idx = np.argsort(ask_prices_slice)

        # 应用排序
        ob_ask_prices[snap_idx, :depth] = ask_prices_slice[ask_sort_idx]
        ob_ask_amounts[snap_idx, :depth] = ask_amounts_slice[ask_sort_idx]

        # Bid侧排序（降序）
        # 提取有效数据
        bid_prices_slice = ob_bid_prices[snap_idx, :depth].copy()
        bid_amounts_slice = ob_bid_amounts[snap_idx, :depth].copy()

        # 降序排序：先升序再反转
        bid_sort_idx = np.argsort(bid_prices_slice)[::-1]

        # 应用排序
        ob_bid_prices[snap_idx, :depth] = bid_prices_slice[bid_sort_idx]
        ob_bid_amounts[snap_idx, :depth] = bid_amounts_slice[bid_sort_idx]


def prepare_orderbook_arrays_optimized(
    orderbook_df,
    max_depth=100
):
    """
    高层封装：将orderbook DataFrame转换为numpy数组（性能优化版）

    使用numba加速，相比pandas groupby版本提速约5倍

    参数：
        orderbook_df: DataFrame，包含字段：
            - timestamp: int64，毫秒时间戳
            - side: str，'bid' or 'ask'
            - price: float64
            - amount: float64
        max_depth: 最大深度，默认100

    返回：
        dict，包含：
            - ob_timestamps: 1D数组，shape=(n_snapshots,)
            - ob_ask_prices: 2D数组，shape=(n_snapshots, max_depth)
            - ob_ask_amounts: 2D数组
            - ob_bid_prices: 2D数组
            - ob_bid_amounts: 2D数组
            - ob_depths: 1D数组，每个截面的实际深度
    """
    import time
    start_t = time.time()

    # 1. 提取唯一时间戳（升序）
    unique_ts = np.sort(orderbook_df['timestamp'].unique())
    n_snapshots = len(unique_ts)

    # 2. 将side字段编码为int8（0=bid, 1=ask）
    side_map = {'bid': 0, 'ask': 1}
    sides_encoded = orderbook_df['side'].map(side_map).values.astype(np.int8)

    # 3. 提取其他字段
    timestamps = orderbook_df['timestamp'].values.astype(np.int64)
    prices = orderbook_df['price'].values.astype(np.float64)
    amounts = orderbook_df['amount'].values.astype(np.float64)

    # 4. 调用numba加速函数
    (ob_ask_prices, ob_ask_amounts,
     ob_bid_prices, ob_bid_amounts, ob_depths) = prepare_orderbook_arrays_numba(
        timestamps,
        sides_encoded,
        prices,
        amounts,
        unique_ts,
        max_depth
    )

    # 5. 排序（ask升序，bid降序）
    sort_orderbook_sides_numba(
        ob_ask_prices,
        ob_ask_amounts,
        ob_bid_prices,
        ob_bid_amounts,
        ob_depths
    )

    elapsed = time.time() - start_t
    print(f"  [性能] orderbook数组转换耗时（numba优化版）: {elapsed:.3f}秒")

    return {
        'ob_timestamps': unique_ts,
        'ob_ask_prices': ob_ask_prices,
        'ob_ask_amounts': ob_ask_amounts,
        'ob_bid_prices': ob_bid_prices,
        'ob_bid_amounts': ob_bid_amounts,
        'ob_depths': ob_depths
    }
