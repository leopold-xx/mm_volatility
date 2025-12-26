# -*- coding: utf-8 -*-
"""
Guard价位计算核心
基于逐笔成交构造悲观的价位边界
"""

import numpy as np
from numba import jit, int64, float64, boolean
import math


@jit(nopython=True)
def compute_guard_batch(timestamps: np.ndarray,
                       prices: np.ndarray,
                       is_buyer_maker: np.ndarray,
                       k_window: int64 = 100,
                       ticksize: float64 = 0.01) -> tuple:
    """
    批量计算Guard价位（numba加速版本）

    基于影子一档（Shadow Quotes）概念：
    - 买方主动单（is_buyer_maker=False）：成交价视为shadow_ask观测
    - 卖方主动单（is_buyer_maker=True）：成交价视为shadow_bid观测

    Guard定义：
    - guard_ask = min(最近K次shadow_ask观测)
    - guard_bid = max(最近K次shadow_bid观测)

    参数：
        timestamps: 时间戳数组（毫秒）
        prices: 成交价格数组
        is_buyer_maker: 买方是否为maker的布尔数组
        k_window: 窗口大小K
        ticksize: 价格最小单位

    返回：
        (guard_ask数组, guard_bid数组)
    """
    n = len(timestamps)

    # 初始化输出数组
    guard_ask = np.zeros(n, dtype=np.float64)
    guard_bid = np.zeros(n, dtype=np.float64)

    # 初始化循环buffer存储最近K笔观测
    # 使用inf/-inf初始化，便于min/max计算
    shadow_ask_buffer = np.full(k_window, np.inf, dtype=np.float64)
    shadow_bid_buffer = np.full(k_window, -np.inf, dtype=np.float64)

    # buffer的写入位置索引
    ask_write_idx = 0
    bid_write_idx = 0

    # 记录实际观测数量（处理初始阶段）
    ask_count = 0
    bid_count = 0

    # 记录上一个有效的guard值（用于兜底）
    last_guard_ask = 0.0
    last_guard_bid = 0.0

    for i in range(n):
        current_price = prices[i]
        is_maker = is_buyer_maker[i]

        # 更新shadow观测
        if not is_maker:  # 买方是taker -> 买单打穿ask -> shadow_ask观测
            # 更新buffer
            shadow_ask_buffer[ask_write_idx % k_window] = current_price
            ask_write_idx += 1
            ask_count = min(ask_count + 1, k_window)

            # 计算guard_ask（取最小值）
            if ask_count > 0:
                # 只考虑有效的观测（前ask_count个）
                valid_count = min(ask_count, k_window)
                min_ask = np.inf
                for j in range(valid_count):
                    val = shadow_ask_buffer[j]
                    if val < min_ask:
                        min_ask = val
                last_guard_ask = min_ask
            else:
                last_guard_ask = current_price  # 兜底

        if is_maker:  # 买方是maker -> 卖单打穿bid -> shadow_bid观测
            # 更新buffer
            shadow_bid_buffer[bid_write_idx % k_window] = current_price
            bid_write_idx += 1
            bid_count = min(bid_count + 1, k_window)

            # 计算guard_bid（取最大值）
            if bid_count > 0:
                # 只考虑有效的观测（前bid_count个）
                valid_count = min(bid_count, k_window)
                max_bid = -np.inf
                for j in range(valid_count):
                    val = shadow_bid_buffer[j]
                    if val > max_bid:
                        max_bid = val
                last_guard_bid = max_bid
            else:
                last_guard_bid = current_price  # 兜底

        # 设置当前guard值
        # 如果没有观测，使用当前价格（需要ticksize对齐）
        if last_guard_ask == 0.0 or last_guard_ask == np.inf:
            # 兜底情况：使用当前价格，向上取整（更严格的ask上界）
            guard_ask[i] = math.ceil(current_price / ticksize) * ticksize
        else:
            # 向下取整到ticksize（更严格的ask上界）
            guard_ask[i] = math.floor(last_guard_ask / ticksize) * ticksize

        if last_guard_bid == 0.0 or last_guard_bid == -np.inf:
            # 兜底情况：使用当前价格，向下取整（更严格的bid下界）
            guard_bid[i] = math.floor(current_price / ticksize) * ticksize
        else:
            # 向上取整到ticksize（更严格的bid下界）
            guard_bid[i] = math.ceil(last_guard_bid / ticksize) * ticksize

        # 确保guard_bid <= guard_ask（基本合理性）
        if guard_bid[i] > guard_ask[i]:
            # 使用当前价格作为中间值
            guard_ask[i] = np.ceil(current_price / ticksize) * ticksize
            guard_bid[i] = np.floor(current_price / ticksize) * ticksize

    return guard_ask, guard_bid


@jit(nopython=True)
def compute_guard_incremental(shadow_ask_buffer: np.ndarray,
                             shadow_bid_buffer: np.ndarray,
                             ask_write_idx: int64,
                             bid_write_idx: int64,
                             ask_count: int64,
                             bid_count: int64,
                             new_price: float64,
                             is_buyer_maker: boolean,
                             k_window: int64,
                             ticksize: float64) -> tuple:
    """
    增量更新Guard价位（用于实时计算）

    参数：
        shadow_ask_buffer: ask观测buffer
        shadow_bid_buffer: bid观测buffer
        ask_write_idx: ask buffer写入位置
        bid_write_idx: bid buffer写入位置
        ask_count: ask观测计数
        bid_count: bid观测计数
        new_price: 新成交价格
        is_buyer_maker: 新成交的买方是否为maker
        k_window: 窗口大小
        ticksize: 价格最小单位

    返回：
        (新guard_ask, 新guard_bid, 更新后的索引和计数)
    """
    guard_ask = 0.0
    guard_bid = 0.0

    # 更新shadow观测
    if not is_buyer_maker:  # shadow_ask观测
        shadow_ask_buffer[ask_write_idx % k_window] = new_price
        ask_write_idx += 1
        ask_count = min(ask_count + 1, k_window)

        # 计算新的guard_ask
        if ask_count > 0:
            valid_count = min(ask_count, k_window)
            min_ask = np.inf
            for j in range(valid_count):
                val = shadow_ask_buffer[j]
                if val < min_ask:
                    min_ask = val
            guard_ask = np.floor(min_ask / ticksize) * ticksize
        else:
            guard_ask = new_price

    if is_buyer_maker:  # shadow_bid观测
        shadow_bid_buffer[bid_write_idx % k_window] = new_price
        bid_write_idx += 1
        bid_count = min(bid_count + 1, k_window)

        # 计算新的guard_bid
        if bid_count > 0:
            valid_count = min(bid_count, k_window)
            max_bid = -np.inf
            for j in range(valid_count):
                val = shadow_bid_buffer[j]
                if val > max_bid:
                    max_bid = val
            guard_bid = np.ceil(max_bid / ticksize) * ticksize
        else:
            guard_bid = new_price

    return (guard_ask, guard_bid, ask_write_idx, bid_write_idx,
            ask_count, bid_count)


@jit(nopython=True)
def check_guard_fail(order_price: float64,
                    order_side: int64,
                    guard_ask: float64,
                    guard_bid: float64) -> boolean:
    """
    检查订单是否违反Guard条件

    规则：
    - 买单价格 >= guard_ask → fail
    - 卖单价格 <= guard_bid → fail

    参数：
        order_price: 订单价格
        order_side: 订单方向（0=买，1=卖）
        guard_ask: 当前guard_ask
        guard_bid: 当前guard_bid

    返回：
        是否fail
    """
    if order_side == 0:  # BUY
        # 买单价格必须严格小于guard_ask
        return order_price >= guard_ask
    else:  # SELL
        # 卖单价格必须严格大于guard_bid
        return order_price <= guard_bid


class GuardCalculator:
    """
    Guard价位计算器（高层封装）

    提供易用的接口和参数管理
    """

    def __init__(self, k_window: int = 100):
        """
        初始化Guard计算器

        参数：
            k_window: 窗口大小K（默认100笔）
        """
        self.k_window = k_window
        print(f"📊 Guard计算器初始化")
        print(f"   窗口大小: K={k_window}")

    def calculate(self, trades_df, ticksize: float = 0.01):
        """
        计算Guard价位

        参数：
            trades_df: 包含trades数据的DataFrame
            ticksize: 价格最小单位

        返回：
            包含guard_ask和guard_bid列的DataFrame
        """
        # 提取必要字段
        timestamps = trades_df['timestamp'].values if 'timestamp' in trades_df.columns else trades_df['transact_time'].values
        prices = trades_df['trade_price'].values if 'trade_price' in trades_df.columns else trades_df['price'].values
        is_buyer_maker = trades_df['is_buyer_maker'].values

        # 调用numba函数计算
        guard_ask, guard_bid = compute_guard_batch(
            timestamps.astype(np.int64),
            prices.astype(np.float64),
            is_buyer_maker,
            self.k_window,
            ticksize
        )

        # 添加到DataFrame
        result_df = trades_df.copy()
        result_df['guard_ask'] = guard_ask
        result_df['guard_bid'] = guard_bid

        # 打印统计信息
        print(f"✅ Guard计算完成")
        print(f"   guard_ask范围: {guard_ask.min():.4f} - {guard_ask.max():.4f}")
        print(f"   guard_bid范围: {guard_bid.min():.4f} - {guard_bid.max():.4f}")
        print(f"   平均价差: {(guard_ask - guard_bid).mean():.4f}")

        return result_df


def get_default_parameters():
    """
    获取默认参数配置

    返回：
        默认参数字典
    """
    return {
        'k_window': 100,  # 默认100笔窗口
    }