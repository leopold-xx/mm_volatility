# -*- coding: utf-8 -*-
"""
完全numba加速的回测引擎
主循环使用numba JIT实现极致性能
"""

import numpy as np
import pandas as pd
import math
from numba import jit, int64, float64, int8
from typing import Optional, Callable

from .order import (
    ORDER_FIELD_COUNT, ORDER_ID_IDX, ORDER_PRICE_IDX, ORDER_QUANTITY_IDX,
    ORDER_SIDE_IDX, ORDER_STATUS_IDX, ORDER_SIDE_BUY, ORDER_SIDE_SELL,
    ORDER_STATUS_ACTIVE, ORDER_STATUS_FILLED, ORDER_STATUS_CANCELLED, ORDER_STATUS_FAILED,
    ORDER_ARRIVE_TIME_IDX, ORDER_STATUS_PENDING, ORDER_STATUS_CANCELLING, ORDER_FILL_TIME_IDX,
    ORDER_BIRTH_TIME_IDX, ORDER_CANCEL_TIME_IDX, ORDER_EX_CANCEL_TIME_IDX, ORDER_FILLED_QTY_IDX,
    create_order_array, batch_update_orders_status,
    initiate_order_cancel, update_position_and_cash, check_guard_fail,
    calculate_price_deviation
)


@jit(nopython=True, cache=True, inline='always')
def create_soa_orders(max_orders: int64):
    """
    创建SoA（Structure of Arrays）格式的订单存储
    优化内存带宽和缓存效率
    """
    # 使用压缩的数据类型
    order_ids = np.zeros(max_orders, dtype=np.int64)        # 订单ID
    prices = np.zeros(max_orders, dtype=np.float64)         # 价格
    quantities = np.zeros(max_orders, dtype=np.float64)     # 数量
    sides = np.zeros(max_orders, dtype=np.int8)             # 方向（0/1）
    statuses = np.zeros(max_orders, dtype=np.int8)          # 状态（0-5）
    birth_times = np.zeros(max_orders, dtype=np.int64)      # 创建时间
    arrive_times = np.zeros(max_orders, dtype=np.int64)     # 到达时间
    cancel_times = np.full(max_orders, -1, dtype=np.int64)  # 撤单时间
    ex_cancel_times = np.full(max_orders, -1, dtype=np.int64)  # 撤单生效时间
    fill_times = np.full(max_orders, -1, dtype=np.int64)    # 成交时间
    filled_qtys = np.zeros(max_orders, dtype=np.float64)    # 已成交数量

    return (order_ids, prices, quantities, sides, statuses,
            birth_times, arrive_times, cancel_times, ex_cancel_times,
            fill_times, filled_qtys)


@jit(nopython=True, cache=True, inline='always')
def aos_to_soa_order(aos_order: np.ndarray, soa_orders: tuple, idx: int64):
    """
    将AoS格式的订单转换为SoA格式并存储到指定位置
    """
    order_ids, prices, quantities, sides, statuses, birth_times, arrive_times, cancel_times, ex_cancel_times, fill_times, filled_qtys = soa_orders

    order_ids[idx] = int64(aos_order[ORDER_ID_IDX])
    prices[idx] = aos_order[ORDER_PRICE_IDX]
    quantities[idx] = aos_order[ORDER_QUANTITY_IDX]
    sides[idx] = int8(aos_order[ORDER_SIDE_IDX])
    statuses[idx] = int8(aos_order[ORDER_STATUS_IDX])
    birth_times[idx] = int64(aos_order[ORDER_BIRTH_TIME_IDX])
    arrive_times[idx] = int64(aos_order[ORDER_ARRIVE_TIME_IDX])
    cancel_times[idx] = int64(aos_order[ORDER_CANCEL_TIME_IDX])
    ex_cancel_times[idx] = int64(aos_order[ORDER_EX_CANCEL_TIME_IDX])
    fill_times[idx] = int64(aos_order[ORDER_FILL_TIME_IDX])
    filled_qtys[idx] = aos_order[ORDER_FILLED_QTY_IDX]


@jit(nopython=True, cache=True, inline='always')
def soa_to_aos_order(soa_orders: tuple, idx: int64, aos_order: np.ndarray):
    """
    将SoA格式的订单转换为AoS格式
    """
    order_ids, prices, quantities, sides, statuses, birth_times, arrive_times, cancel_times, ex_cancel_times, fill_times, filled_qtys = soa_orders

    aos_order[ORDER_ID_IDX] = float64(order_ids[idx])
    aos_order[ORDER_PRICE_IDX] = prices[idx]
    aos_order[ORDER_QUANTITY_IDX] = quantities[idx]
    aos_order[ORDER_SIDE_IDX] = float64(sides[idx])
    aos_order[ORDER_STATUS_IDX] = float64(statuses[idx])
    aos_order[ORDER_BIRTH_TIME_IDX] = float64(birth_times[idx])
    aos_order[ORDER_ARRIVE_TIME_IDX] = float64(arrive_times[idx])
    aos_order[ORDER_CANCEL_TIME_IDX] = float64(cancel_times[idx])
    aos_order[ORDER_EX_CANCEL_TIME_IDX] = float64(ex_cancel_times[idx])
    aos_order[ORDER_FILL_TIME_IDX] = float64(fill_times[idx])
    aos_order[ORDER_FILLED_QTY_IDX] = filled_qtys[idx]


@jit(nopython=True, cache=True)
def make_market_decision_fast(live_order_indices: np.ndarray,
                             live_order_count: int64,
                             all_orders: np.ndarray,
                             next_order_id: int64,
                             position: float64,
                             current_time: int64,
                             ask_permit: float64,
                             bid_permit: float64,
                             price_tolerance: float64,
                             initial_equity: float64,  # 改为初始权益
                             place_fixed_latency: int64,
                             place_random_min: int64,
                             place_random_max: int64,
                             cancel_fixed_latency: int64,
                             cancel_random_min: int64,
                             cancel_random_max: int64,
                             new_orders_buf: np.ndarray,
                             cancel_ids_buf: np.ndarray) -> tuple:
    """
    决策分离式做市决策函数（numba加速）

    核心逻辑：
    1. 决策分离：撤单和挂单决策完全独立
    2. 仓位优先：有仓位时优先考虑平仓
    3. 固定风险敞口：无仓位时使用固定的初始权益
    4. 先撤后挂：数量不匹配时先撤单，下次循环重新挂单

    撤单条件（满足任一条件即撤单）：
    - 仓位约束：持仓方向不能有订单
    - 价格偏差：超出设定容差范围
    - 数量不匹配：有仓位时，现有订单数量与持仓量不符

    挂单数量计算：
    - 空仓时：order_size = initial_equity / order_price
    - 有仓位时：order_size = abs(position)

    返回：
        (new_orders_count, cancel_orders_count, 更新后的next_order_id)
    """
    new_orders_count = 0
    cancel_orders_count = 0

    # 统计存活订单方向
    has_bid_orders = False
    has_ask_orders = False

    # 第一步：撤单决策（只检查存活订单）
    for i in range(live_order_count):
        if i >= len(live_order_indices) or live_order_indices[i] < 0:
            continue

        order_idx = live_order_indices[i]
        order = all_orders[order_idx]
        order_side = int64(order[ORDER_SIDE_IDX])
        order_status = int64(order[ORDER_STATUS_IDX])

        # 只检查真正存活的订单
        if order_status not in (ORDER_STATUS_PENDING, ORDER_STATUS_ACTIVE, ORDER_STATUS_CANCELLING):
            continue

        if order_side == ORDER_SIDE_BUY:
            has_bid_orders = True
            # 买单撤单条件
            should_cancel = False

            # 条件1：仓位约束 - 多头仓位时买方不能有订单
            if position > 0.0:
                should_cancel = True
            else:
                existing_price = order[ORDER_PRICE_IDX]
                existing_quantity = order[ORDER_QUANTITY_IDX]

                # 条件2：价格偏差检查
                deviation = calculate_price_deviation(existing_price, bid_permit)
                if deviation > price_tolerance:
                    should_cancel = True
                # 条件3：数量不匹配检查（仅在空头仓位且价格合适时）
                elif position < 0.0:
                    required_quantity = abs(position)  # 空头仓位需要的平仓数量
                    if abs(existing_quantity - required_quantity) > 1e-10:
                        should_cancel = True

            if should_cancel and cancel_orders_count < len(cancel_ids_buf):
                cancel_ids_buf[cancel_orders_count] = int64(order[ORDER_ID_IDX])
                cancel_orders_count += 1

        else:  # ORDER_SIDE_SELL
            has_ask_orders = True
            # 卖单撤单条件
            should_cancel = False

            # 条件1：仓位约束 - 空头仓位时卖方不能有订单
            if position < 0.0:
                should_cancel = True
            else:
                existing_price = order[ORDER_PRICE_IDX]
                existing_quantity = order[ORDER_QUANTITY_IDX]

                # 条件2：价格偏差检查
                deviation = calculate_price_deviation(existing_price, ask_permit)
                if deviation > price_tolerance:
                    should_cancel = True
                # 条件3：数量不匹配检查（仅在多头仓位且价格合适时）
                elif position > 0.0:
                    required_quantity = position  # 多头仓位需要的平仓数量
                    if abs(existing_quantity - required_quantity) > 1e-10:
                        should_cancel = True

            if should_cancel and cancel_orders_count < len(cancel_ids_buf):
                cancel_ids_buf[cancel_orders_count] = int64(order[ORDER_ID_IDX])
                cancel_orders_count += 1

    # 第二步：挂单决策（只有当该侧完全没有存续订单时）
    # 买方挂单决策
    if not has_bid_orders and position <= 0.0 and new_orders_count < len(new_orders_buf):
        # 计算买单数量
        if position == 0.0:
            # 空仓：使用初始权益建立新仓位
            bid_order_size = initial_equity / bid_permit if bid_permit > 0 else 0.0
        else:  # position < 0.0
            # 空头仓位：使用持仓量平仓
            bid_order_size = abs(position)

        if bid_order_size > 0:
            bid_order = create_order_array(
                next_order_id, bid_permit, bid_order_size, ORDER_SIDE_BUY,
                current_time, place_fixed_latency, place_random_min, place_random_max
            )
            new_orders_buf[new_orders_count] = bid_order
            new_orders_count += 1
            next_order_id += 1

    # 卖方挂单决策
    if not has_ask_orders and position >= 0.0 and new_orders_count < len(new_orders_buf):
        # 计算卖单数量
        if position == 0.0:
            # 空仓：使用初始权益建立新仓位
            ask_order_size = initial_equity / ask_permit if ask_permit > 0 else 0.0
        else:  # position > 0.0
            # 多头仓位：使用持仓量平仓
            ask_order_size = position

        if ask_order_size > 0:
            ask_order = create_order_array(
                next_order_id, ask_permit, ask_order_size, ORDER_SIDE_SELL,
                current_time, place_fixed_latency, place_random_min, place_random_max
            )
            new_orders_buf[new_orders_count] = ask_order
            new_orders_count += 1
            next_order_id += 1

    return (new_orders_count, cancel_orders_count, next_order_id)


@jit(nopython=True, cache=True)
def make_market_decision_with_penalty(live_order_indices: np.ndarray,
                                      live_order_count: int64,
                                      all_orders: np.ndarray,
                                      next_order_id: int64,
                                      position: float64,
                                      current_time: int64,
                                      ask_permit: float64,
                                      bid_permit: float64,
                                      pema: float64,  # 新增：公允价格参数
                                      pos_punish: float64,  # 新增：仓位惩罚因子(0.0~1.0)
                                      risk_off: bool,          # NEW
                                      price_tolerance: float64,
                                      initial_equity: float64,
                                      place_fixed_latency: int64,
                                      place_random_min: int64,
                                      place_random_max: int64,
                                      cancel_fixed_latency: int64,
                                      cancel_random_min: int64,
                                      cancel_random_max: int64,
                                      new_orders_buf: np.ndarray,
                                      cancel_ids_buf: np.ndarray) -> tuple:
    """
    带仓位惩罚的做市决策函数（numba加速）

    核心逻辑：
    1. 决策分离：撤单和挂单决策完全独立
    2. 仓位优先：有仓位时优先考虑平仓
    3. 固定风险敞口：无仓位时使用固定的初始权益
    4. 仓位惩罚：持仓时将平仓订单价格向公允价格靠拢

    仓位惩罚公式：
    - 多仓平仓（卖单）：adjusted_ask = ask_permit - (ask_permit - pema) * pos_punish
    - 空仓平仓（买单）：adjusted_bid = bid_permit + (pema - bid_permit) * pos_punish

    pos_punish参数：
    - 0.0：不做惩罚，等同于原始策略
    - 0.5：订单价格为ask_permit/bid_permit与pema的中点
    - 1.0：订单直接挂在公允价格pema

    返回：
        (new_orders_count, cancel_orders_count, 更新后的next_order_id)
    """
    new_orders_count = 0
    cancel_orders_count = 0

    # 统计存活订单方向
    has_bid_orders = False
    has_ask_orders = False

    # NEW：只在 risk_off 用，用来判断“撤完后是否还留着某侧单”
    kept_bid = False
    kept_ask = False

    # 第一步：撤单决策（逻辑与原始策略完全相同）
    for i in range(live_order_count):
        if i >= len(live_order_indices) or live_order_indices[i] < 0:
            continue

        order_idx = live_order_indices[i]
        order = all_orders[order_idx]
        order_side = int64(order[ORDER_SIDE_IDX])
        order_status = int64(order[ORDER_STATUS_IDX])

        # 只检查真正存活的订单
        if order_status not in (ORDER_STATUS_PENDING, ORDER_STATUS_ACTIVE, ORDER_STATUS_CANCELLING):
            continue
        
        if order_side == ORDER_SIDE_BUY:
            has_bid_orders = True
        else:
            has_ask_orders = True

        should_cancel = False

        # risk_off 额外约束（在进入原逻辑前先裁决哪些必须撤）
        if risk_off:
            if position == 0.0:
                # 空仓：risk_off 不允许任何挂单
                should_cancel = True
            elif position > 0.0:
                # 多仓：risk_off 不允许 BUY（避免加仓/对冲）
                if order_side == ORDER_SIDE_BUY:
                    should_cancel = True
            else:
                # 空仓：risk_off 不允许 SELL
                if order_side == ORDER_SIDE_SELL:
                    should_cancel = True

        # 如果 risk_off 已经判定必须撤，则不用再跑原逻辑（但也不影响原逻辑）以下是原始逻辑
        if not should_cancel:
            if order_side == ORDER_SIDE_BUY:
                # 条件1：仓位约束 - 多头仓位时买方不能有订单
                if position > 0.0:
                    should_cancel = True
                else:
                    existing_price = order[ORDER_PRICE_IDX]
                    existing_quantity = order[ORDER_QUANTITY_IDX]

                    # 计算目标价格（应用仓位惩罚）
                    if position < 0.0:
                        # 空仓平仓：买单向pema靠拢
                        target_bid = bid_permit + (pema - bid_permit) * pos_punish
                    else:
                        # 空仓建仓：使用原始bid_permit
                        target_bid = bid_permit

                    # 条件2：价格偏差检查（与目标价格比较）
                    deviation = calculate_price_deviation(existing_price, target_bid)
                    if deviation > price_tolerance:
                        should_cancel = True
                    # 条件3：数量不匹配检查（仅在空头仓位且价格合适时）
                    elif position < 0.0:
                        required_quantity = abs(position)  # 空头仓位需要的平仓数量
                        if abs(existing_quantity - required_quantity) > 1e-10:
                            should_cancel = True

            else:  # ORDER_SIDE_SELL

                # 条件1：仓位约束 - 空头仓位时卖方不能有订单
                if position < 0.0:
                    should_cancel = True
                else:
                    existing_price = order[ORDER_PRICE_IDX]
                    existing_quantity = order[ORDER_QUANTITY_IDX]

                    # 计算目标价格（应用仓位惩罚）
                    if position > 0.0:
                        # 多仓平仓：卖单向pema靠拢
                        target_ask = ask_permit - (ask_permit - pema) * pos_punish
                    else:
                        # 空仓建仓：使用原始ask_permit
                        target_ask = ask_permit

                    # 条件2：价格偏差检查（与目标价格比较）
                    deviation = calculate_price_deviation(existing_price, target_ask)
                    if deviation > price_tolerance:
                        should_cancel = True
                    # 条件3：数量不匹配检查（仅在多头仓位且价格合适时）
                    elif position > 0.0:
                        required_quantity = position  # 多头仓位需要的平仓数量
                        if abs(existing_quantity - required_quantity) > 1e-10:
                            should_cancel = True

        if should_cancel and cancel_orders_count < len(cancel_ids_buf):
            cancel_ids_buf[cancel_orders_count] = int64(order[ORDER_ID_IDX])
            cancel_orders_count += 1
        else:
            # NEW：risk_off 下记录“撤完后保留的单”
            if order_side == ORDER_SIDE_BUY:
                kept_bid = True
            else:
                kept_ask = True

    # 第二步：挂单决策（应用仓位惩罚）
    if risk_off:
        # risk_off：只平仓、不建仓、不做市
        if position > 0.0:
            # 多仓：只允许/只需要一个 SELL 平仓单
            if (not kept_ask) and new_orders_count < len(new_orders_buf):
                actual_ask_price = ask_permit - (ask_permit - pema) * pos_punish
                ask_order_size = position
                if ask_order_size > 0:
                    ask_order = create_order_array(
                        next_order_id, actual_ask_price, ask_order_size, ORDER_SIDE_SELL,
                        current_time, place_fixed_latency, place_random_min, place_random_max
                    )
                    new_orders_buf[new_orders_count] = ask_order
                    new_orders_count += 1
                    next_order_id += 1

        elif position < 0.0:
            # 空仓：只允许/只需要一个 BUY 平仓单
            if (not kept_bid) and new_orders_count < len(new_orders_buf):
                actual_bid_price = bid_permit + (pema - bid_permit) * pos_punish
                bid_order_size = abs(position)
                if bid_order_size > 0:
                    bid_order = create_order_array(
                        next_order_id, actual_bid_price, bid_order_size, ORDER_SIDE_BUY,
                        current_time, place_fixed_latency, place_random_min, place_random_max
                    )
                    new_orders_buf[new_orders_count] = bid_order
                    new_orders_count += 1
                    next_order_id += 1

        # position == 0：不挂任何单
        return (new_orders_count, cancel_orders_count, next_order_id)

    # 原始买方挂单决策
    if not has_bid_orders and position <= 0.0 and new_orders_count < len(new_orders_buf):
        # 计算买单价格（应用仓位惩罚）
        if position < 0.0:
            # 空仓平仓：买单向pema靠拢
            actual_bid_price = bid_permit + (pema - bid_permit) * pos_punish
            bid_order_size = abs(position)
        else:  # position == 0.0
            # 空仓建仓：使用原始bid_permit，不应用惩罚
            actual_bid_price = bid_permit
            bid_order_size = initial_equity / bid_permit if bid_permit > 0 else 0.0

        if bid_order_size > 0:
            bid_order = create_order_array(
                next_order_id, actual_bid_price, bid_order_size, ORDER_SIDE_BUY,
                current_time, place_fixed_latency, place_random_min, place_random_max
            )
            new_orders_buf[new_orders_count] = bid_order
            new_orders_count += 1
            next_order_id += 1

    # 原始卖方挂单决策
    if not has_ask_orders and position >= 0.0 and new_orders_count < len(new_orders_buf):
        # 计算卖单价格（应用仓位惩罚）
        if position > 0.0:
            # 多仓平仓：卖单向pema靠拢
            actual_ask_price = ask_permit - (ask_permit - pema) * pos_punish
            ask_order_size = position
        else:  # position == 0.0
            # 空仓建仓：使用原始ask_permit，不应用惩罚
            actual_ask_price = ask_permit
            ask_order_size = initial_equity / ask_permit if ask_permit > 0 else 0.0

        if ask_order_size > 0:
            ask_order = create_order_array(
                next_order_id, actual_ask_price, ask_order_size, ORDER_SIDE_SELL,
                current_time, place_fixed_latency, place_random_min, place_random_max
            )
            new_orders_buf[new_orders_count] = ask_order
            new_orders_count += 1
            next_order_id += 1

    return (new_orders_count, cancel_orders_count, next_order_id)


@jit(nopython=True, cache=True)
def run_backtest_core(trades_timestamp: np.ndarray,
                     trades_price: np.ndarray,
                     trades_qty: np.ndarray,
                     trades_side: np.ndarray,
                     
                     pema: np.ndarray,  # PEMA公允价格数组
                     sigma: np.ndarray,              # NEW: 用于动态 sigma_multi 现算报价

                     guard_ask: np.ndarray,  # Guard价位数组
                     guard_bid: np.ndarray,  # Guard价位数组

                     ask_permit: np.ndarray,  # 预计算的挂单价格
                     bid_permit: np.ndarray,  # 预计算的挂单价格

                     initial_cash: float64,
                     initial_equity: float64,  # 替换order_size为初始权益

                     price_tolerance: float64,  # 容差参数
                     pos_punish: float64,  # 新增：仓位惩罚因子(0.0~1.0)
                     ticksize: float64,               # NEW: ticksize 传进来

                     sigma_multi_arr: np.ndarray,     # NEW: 每笔 sigma_multi
                     price_tol_arr: np.ndarray,       # NEW: 每笔 price_tolerance
                     pos_punish_arr: np.ndarray,      # NEW: 每笔 pos_punish
                     risk_off_arr: np.ndarray,        # NEW: 每笔是否 risk_off
                     
                     use_dynamic_sigma_multi: bool,   # NEW: 是否动态报价
                     use_dynamic_pos_punish: bool,
                     use_dynamic_price_tol: bool,
                     use_risk_off: bool,

                     place_fixed_latency: int64,
                     place_random_min: int64,
                     place_random_max: int64,
                     cancel_fixed_latency: int64,
                     cancel_random_min: int64,
                     cancel_random_max: int64,
                     output_mode: int64,  # 0=detail, 1=summary
                     aggregation_seconds: int64) -> tuple:  # 聚合周期（秒）
    """
    完全numba加速的回测核心循环

    关键修正：
    1. 移除max_orders限制，保存所有订单数据
    2. 优化存活订单管理，仅检查存活订单状态
    3. 使用初始权益动态计算下单量，保持固定风险敞口
    4. 完整保存所有订单生命周期数据
    5. 支持仓位惩罚策略(pos_punish > 0时启用)

    返回：
        (final_cash, final_position, all_orders_data, fill_history_data)
    """
    # 初始化账户状态
    cash = initial_cash
    position = 0.0
    next_order_id = 1

    # 统计信息
    fill_count = 0
    guard_fail_count = 0
    trade_count = len(trades_timestamp)

    # 存活订单管理：维护存活订单的索引集合
    # 由于numba限制，使用固定大小数组模拟集合，最多同时存在2个存活订单
    max_live_orders = 2  # 策略设计：最多一买一卖
    live_order_indices = np.full(max_live_orders, -1, dtype=np.int64)  # -1表示空位
    live_order_count = 0

    # 所有订单数据存储（动态扩展的概念，但numba需要预分配）
    # 提升容量到安全上界，避免溢出（每2个trade最多一个订单，再乘以2倍安全系数）
    estimated_max_orders = max(trade_count, 1000)  # 保守且安全的估计
    all_orders = np.zeros((estimated_max_orders, ORDER_FIELD_COUNT), dtype=np.float64)
    total_order_count = 0

    # 完整订单历史记录（无限制）
    order_history = np.zeros((estimated_max_orders, ORDER_FIELD_COUNT + 2), dtype=np.float64)
    # 额外2列：create_time, final_time（成交/撤单/失败时间）
    order_history_count = 0

    # 持仓时间序列记录（记录每次持仓变化）
    # 预分配数组，最多trade_count次持仓变化
    max_position_changes = trade_count
    position_history = np.zeros((max_position_changes, 4), dtype=np.float64)
    position_history_count = 0

    # 逐笔trade完整状态记录（记录每个trade的完整状态）
    # 列定义：0:trade_idx, 1:timestamp, 2:trade_price, 3:trade_qty, 4:trade_side
    # 5:pema, 6:ask_permit, 7:bid_permit, 8:guard_ask, 9:guard_bid
    # 10:cash, 11:position, 12:total_value
    # 13:buy_orders_alive, 14:sell_orders_alive, 15:decision_flag
    # output_mode: 0=detail(逐笔记录), 1=summary(秒级K线)
    if output_mode == 0:  # detail模式
        trade_history = np.zeros((trade_count, 16), dtype=np.float64)
    else:  # summary模式，预分配大缓冲区，最后会裁剪
        # 根据聚合周期估算bar数量
        time_range_ms = trades_timestamp[-1] - trades_timestamp[0]
        estimated_bars = int(time_range_ms / (aggregation_seconds * 1000)) + 100
        # 字段：0:timestamp, 1:datetime, 2:open, 3:high, 4:low, 5:close
        # 6:volume, 7:turnover, 8:buy_count, 9:sell_count
        # 10:cash, 11:position, 12:total_value, 13:bar_pnl, 14:cumulative_pnl
        # 15:pema, 16:ask_permit, 17:bid_permit
        # 18:orders_placed, 19:orders_filled, 20:buy_orders, 21:sell_orders
        # 22:buy_fills, 23:sell_fills, 24:filled_volume, 25:filled_turnover
        # 26:long_close_profit, 27:short_close_profit
        trade_history = np.zeros((estimated_bars, 28), dtype=np.float64)

    # summary模式的K线聚合变量
    current_period = -1
    bar_idx = -1
    bar_open = 0.0
    bar_high = 0.0
    bar_low = 0.0
    bar_close = 0.0
    bar_volume = 0.0
    bar_turnover = 0.0
    bar_buy_count = 0
    bar_sell_count = 0

    # 新增：账户成交统计变量
    bar_filled_volume = 0.0
    bar_filled_turnover = 0.0

    # 上一bar总权益（用于计算bar_pnl）
    prev_total_value = initial_cash

    # 新增：盈利计算状态变量
    long_open_price = 0.0    # 多头开仓价格
    short_open_price = 0.0   # 空头开仓价格

    # 新增：周期内计数器（每个聚合周期重置）
    period_orders_placed = 0
    period_orders_filled = 0
    period_buy_orders = 0
    period_sell_orders = 0
    period_buy_fills = 0
    period_sell_fills = 0
    period_long_close_profit = 0.0
    period_short_close_profit = 0.0

    # 预分配决策函数使用的缓冲区（避免循环内分配）
    new_orders_buf = np.zeros((2, ORDER_FIELD_COUNT), dtype=np.float64)  # 最多买卖各一个
    cancel_ids_buf = np.zeros(2, dtype=np.int64)  # 最多撤销2个订单

    # 主循环：逐个处理trades
    for trade_idx in range(trade_count):
        current_time = trades_timestamp[trade_idx]
        trade_price = trades_price[trade_idx]
        trade_qty = trades_qty[trade_idx]
        trade_side = trades_side[trade_idx]

        # 1. 检查成交条件（在状态更新之前，优先级最高）
        for i in range(max_live_orders):
            if live_order_indices[i] >= 0:
                order_idx = live_order_indices[i]
                order = all_orders[order_idx]

                # 检查单个订单成交条件（只有ACTIVE或CANCELLING状态的订单才能被撮合）
                order_status = int64(order[ORDER_STATUS_IDX])
                if order_status in (ORDER_STATUS_ACTIVE, ORDER_STATUS_CANCELLING):
                    order_side = int64(order[ORDER_SIDE_IDX])
                    order_price = order[ORDER_PRICE_IDX]

                    # 撮合条件（最悲观估计）
                    filled = False
                    if order_side == ORDER_SIDE_BUY and trade_price < order_price:
                        filled = True
                    elif order_side == ORDER_SIDE_SELL and trade_price > order_price:
                        filled = True

                    if filled:
                        # 订单成交（设置最终状态和成交时间）
                        order[ORDER_STATUS_IDX] = float64(ORDER_STATUS_FILLED)
                        order[ORDER_FILL_TIME_IDX] = float64(current_time)  # 设置成交时间

                        # 更新持仓和资金
                        old_position = position
                        cash, position = update_position_and_cash(cash, position, order)
                        fill_count += 1

                        # 盈利计算和成交统计
                        order_qty = order[ORDER_QUANTITY_IDX]

                        # 统计成交数和账户成交量/成交额
                        period_orders_filled += 1
                        if order_side == ORDER_SIDE_BUY:
                            period_buy_fills += 1
                        else:
                            period_sell_fills += 1

                        # 累加账户成交统计
                        bar_filled_volume += order_qty
                        bar_filled_turnover += order_price * order_qty

                        # 检测开仓事件（从0到非0）
                        if old_position == 0.0 and position > 0.0:
                            long_open_price = order_price  # 记录多头开仓价
                        elif old_position == 0.0 and position < 0.0:
                            short_open_price = order_price  # 记录空头开仓价

                        # 检测平仓事件（从非0到0）
                        if old_position > 0.0 and position == 0.0:
                            # 平多盈利 = (平仓价 - 开仓价) × 持仓量
                            profit = (order_price - long_open_price) * old_position
                            period_long_close_profit += profit
                        elif old_position < 0.0 and position == 0.0:
                            # 平空盈利 = (开仓价 - 平仓价) × 持仓量绝对值
                            profit = (short_open_price - order_price) * abs(old_position)
                            period_short_close_profit += profit

                        # 记录持仓变化（如果持仓发生变化）
                        if position != old_position and position_history_count < max_position_changes:
                            total_value = cash + position * trade_price
                            position_history[position_history_count, 0] = float64(current_time)
                            position_history[position_history_count, 1] = cash
                            position_history[position_history_count, 2] = position
                            position_history[position_history_count, 3] = total_value
                            position_history_count += 1

                        # 记录成交订单到历史
                        for j in range(ORDER_FIELD_COUNT):
                            order_history[order_history_count][j] = all_orders[order_idx][j]
                        order_history[order_history_count][ORDER_FIELD_COUNT] = current_time
                        order_history[order_history_count][ORDER_FIELD_COUNT + 1] = current_time  # final_time
                        order_history_count += 1

                        # 从存活订单集合中移除
                        live_order_indices[i] = -1
                        live_order_count -= 1

        # 2. 更新未成交订单的状态（撤单、失败等）
        for i in range(max_live_orders):
            if live_order_indices[i] >= 0:  # 有存活订单且未成交
                order_idx = live_order_indices[i]

                # 更新订单状态（只有ACTIVE订单可能变为CANCELLED/FAILED）
                batch_update_orders_status(all_orders[order_idx:order_idx+1], 1, current_time)
                new_status = int64(all_orders[order_idx][ORDER_STATUS_IDX])

                # 如果订单状态变为非存活状态，从存活集合中移除
                if new_status in (ORDER_STATUS_CANCELLED, ORDER_STATUS_FAILED):
                    # 记录订单完成信息
                    for j in range(ORDER_FIELD_COUNT):
                        order_history[order_history_count][j] = all_orders[order_idx][j]
                    order_history[order_history_count][ORDER_FIELD_COUNT] = current_time
                    order_history[order_history_count][ORDER_FIELD_COUNT + 1] = current_time  # final_time
                    order_history_count += 1

                    # 从存活订单集合中移除
                    live_order_indices[i] = -1
                    live_order_count -= 1

        # 3. Guard检查（只对刚到达的存活订单）
        current_guard_ask = guard_ask[trade_idx]
        current_guard_bid = guard_bid[trade_idx]

        for i in range(max_live_orders):
            if live_order_indices[i] >= 0:
                order_idx = live_order_indices[i]
                order = all_orders[order_idx]

                # 只检查刚变为ACTIVE的订单
                if (int64(order[ORDER_STATUS_IDX]) == ORDER_STATUS_ACTIVE and
                    current_time == int64(order[ORDER_ARRIVE_TIME_IDX])):
                    if check_guard_fail(order, current_guard_ask, current_guard_bid):
                        order[ORDER_STATUS_IDX] = float64(ORDER_STATUS_FAILED)
                        guard_fail_count += 1

                        # 记录失败订单到历史（重要：确保失败订单也被记录）
                        for j in range(ORDER_FIELD_COUNT):
                            order_history[order_history_count][j] = all_orders[order_idx][j]
                        order_history[order_history_count][ORDER_FIELD_COUNT] = current_time
                        order_history[order_history_count][ORDER_FIELD_COUNT + 1] = current_time  # 失败时间
                        order_history_count += 1

                        # 从存活订单集合中移除
                        live_order_indices[i] = -1
                        live_order_count -= 1

        # 4. 统计当前存活订单数量（在策略决策前）
        buy_alive_count = 0
        sell_alive_count = 0
        for i in range(max_live_orders):
            if live_order_indices[i] >= 0:
                order_idx = live_order_indices[i]
                order_side = int64(all_orders[order_idx][ORDER_SIDE_IDX])
                status = int64(all_orders[order_idx][ORDER_STATUS_IDX])
                if status in (ORDER_STATUS_PENDING, ORDER_STATUS_ACTIVE, ORDER_STATUS_CANCELLING):
                    if order_side == ORDER_SIDE_BUY:
                        buy_alive_count += 1
                    else:
                        sell_alive_count += 1

        # 5. 策略决策点：基于预计算指标的智能做市策略（每个trade都进行决策）
        # === 动态参数（逐笔） ===
        current_pema = pema[trade_idx]
        pos_punish_now = pos_punish_arr[trade_idx] if use_dynamic_pos_punish else pos_punish
        price_tol_now  = price_tol_arr[trade_idx]  if use_dynamic_price_tol else price_tolerance
        risk_off_now   = risk_off_arr[trade_idx] if use_risk_off else False

        # === 动态报价（可选） ===
        if use_dynamic_sigma_multi:
            sm  = sigma_multi_arr[trade_idx]
            sig = sigma[trade_idx]
            current_ask_permit = math.ceil((current_pema + sig * sm) / ticksize) * ticksize
            current_bid_permit = math.floor((current_pema - sig * sm) / ticksize) * ticksize
        else:
            current_ask_permit = ask_permit[trade_idx]
            current_bid_permit = bid_permit[trade_idx]

        # === 统一决策：永远走 with_penalty（pos_punish_now=0 时等价原策略） ===
        new_orders_count, cancel_orders_count, updated_next_order_id = make_market_decision_with_penalty(
            live_order_indices, live_order_count, all_orders, next_order_id, position, current_time,
            current_ask_permit, current_bid_permit,
            current_pema,
            pos_punish_now,
            bool(risk_off_now),
            price_tol_now,
            initial_equity,
            place_fixed_latency, place_random_min, place_random_max,
            cancel_fixed_latency, cancel_random_min, cancel_random_max,
            new_orders_buf, cancel_ids_buf
        )

        # 处理撤单指令（在存活订单中查找）
        for i in range(cancel_orders_count):
            cancel_id = cancel_ids_buf[i]
            # 在存活订单中查找要撤单的订单
            for j in range(max_live_orders):
                if live_order_indices[j] >= 0:
                    order_idx = live_order_indices[j]
                    if int64(all_orders[order_idx][ORDER_ID_IDX]) == cancel_id:
                        initiate_order_cancel(
                            all_orders[order_idx], current_time,
                            cancel_fixed_latency, cancel_random_min, cancel_random_max
                        )
                        break

        # 统计新订单
        for i in range(new_orders_count):
            order_side = int64(new_orders_buf[i][ORDER_SIDE_IDX])
            period_orders_placed += 1
            if order_side == ORDER_SIDE_BUY:
                period_buy_orders += 1
            else:
                period_sell_orders += 1

        # 添加新订单到存活订单集合
        for i in range(new_orders_count):
            if total_order_count < estimated_max_orders:
                # 添加到所有订单数组（所有订单都要记录）
                all_orders[total_order_count] = new_orders_buf[i]

                # 尝试添加到存活订单集合（如果有空位）
                if live_order_count < max_live_orders:
                    for j in range(max_live_orders):
                        if live_order_indices[j] == -1:  # 找到空位
                            live_order_indices[j] = total_order_count
                            live_order_count += 1
                            break
                else:
                    # 存活订单集合已满，订单立即设为失败状态
                    all_orders[total_order_count][ORDER_STATUS_IDX] = float64(ORDER_STATUS_FAILED)

                    # 立即记录到订单历史
                    for j in range(ORDER_FIELD_COUNT):
                        order_history[order_history_count][j] = all_orders[total_order_count][j]
                    order_history[order_history_count][ORDER_FIELD_COUNT] = current_time
                    order_history[order_history_count][ORDER_FIELD_COUNT + 1] = current_time  # final_time
                    order_history_count += 1

                total_order_count += 1

        next_order_id = updated_next_order_id

        # 6. 记录数据（根据输出模式）
        # 计算当前总权益
        total_value = cash + position * trade_price

        # 获取当前技术指标和Guard价位
        current_pema = pema[trade_idx]
        current_ask_permit_val = current_ask_permit
        current_bid_permit_val = current_bid_permit
        current_guard_ask = guard_ask[trade_idx]
        current_guard_bid = guard_bid[trade_idx]

        if output_mode == 0:  # detail模式：逐笔记录
            # 记录完整trade数据
            trade_history[trade_idx][0] = float64(trade_idx)
            trade_history[trade_idx][1] = float64(current_time)
            trade_history[trade_idx][2] = trade_price
            trade_history[trade_idx][3] = trade_qty
            trade_history[trade_idx][4] = float64(trade_side)
            trade_history[trade_idx][5] = current_pema
            trade_history[trade_idx][6] = current_ask_permit_val
            trade_history[trade_idx][7] = current_bid_permit_val
            trade_history[trade_idx][8] = current_guard_ask
            trade_history[trade_idx][9] = current_guard_bid
            trade_history[trade_idx][10] = cash
            trade_history[trade_idx][11] = position
            trade_history[trade_idx][12] = total_value
            trade_history[trade_idx][13] = float64(buy_alive_count)
            trade_history[trade_idx][14] = float64(sell_alive_count)
            # 决策标记：是否产生了新订单或撤单操作
            has_action = 1.0 if (new_orders_count > 0 or cancel_orders_count > 0) else 0.0
            trade_history[trade_idx][15] = has_action

        else:  # summary模式：可配置周期K线聚合
            # 计算当前trade属于哪个聚合周期
            trade_period = int(current_time / (aggregation_seconds * 1000))

            # 如果是新的聚合周期，保存上一周期K线并初始化新K线
            if trade_period != current_period:
                # 保存上一周期的bar（如果存在）
                if current_period >= 0 and bar_idx >= 0:
                    # 计算时间戳（周期开始时间）
                    period_start_ms = current_period * aggregation_seconds * 1000

                    # 字段：0:timestamp, 1:datetime, 2:open, 3:high, 4:low, 5:close
                    trade_history[bar_idx][0] = float64(period_start_ms)
                    trade_history[bar_idx][1] = float64(period_start_ms)  # datetime将在后处理中转换
                    trade_history[bar_idx][2] = bar_open
                    trade_history[bar_idx][3] = bar_high
                    trade_history[bar_idx][4] = bar_low
                    trade_history[bar_idx][5] = bar_close

                    # 市场成交统计
                    trade_history[bar_idx][6] = bar_volume
                    trade_history[bar_idx][7] = bar_turnover
                    trade_history[bar_idx][8] = float64(bar_buy_count)
                    trade_history[bar_idx][9] = float64(bar_sell_count)

                    # 账户状态（周期末）
                    trade_history[bar_idx][10] = cash
                    trade_history[bar_idx][11] = position
                    trade_history[bar_idx][12] = total_value

                    # 盈亏计算
                    bar_pnl = total_value - prev_total_value
                    cumulative_pnl = total_value - initial_cash
                    trade_history[bar_idx][13] = bar_pnl
                    trade_history[bar_idx][14] = cumulative_pnl
                    prev_total_value = total_value

                    # 技术指标（周期末状态）
                    trade_history[bar_idx][15] = current_pema
                    trade_history[bar_idx][16] = current_ask_permit_val
                    trade_history[bar_idx][17] = current_bid_permit_val

                    # 订单统计（周期内累计）
                    trade_history[bar_idx][18] = float64(period_orders_placed)
                    trade_history[bar_idx][19] = float64(period_orders_filled)
                    trade_history[bar_idx][20] = float64(period_buy_orders)
                    trade_history[bar_idx][21] = float64(period_sell_orders)
                    trade_history[bar_idx][22] = float64(period_buy_fills)
                    trade_history[bar_idx][23] = float64(period_sell_fills)

                    # 账户成交统计（周期内累计）
                    trade_history[bar_idx][24] = bar_filled_volume
                    trade_history[bar_idx][25] = bar_filled_turnover

                    # 平仓盈利（周期内累计）
                    trade_history[bar_idx][26] = period_long_close_profit
                    trade_history[bar_idx][27] = period_short_close_profit

                # 初始化新bar
                current_period = trade_period
                bar_idx += 1
                bar_open = trade_price
                bar_high = trade_price
                bar_low = trade_price
                bar_close = trade_price
                bar_volume = 0.0
                bar_turnover = 0.0
                bar_buy_count = 0
                bar_sell_count = 0

                # 重置账户成交统计
                bar_filled_volume = 0.0
                bar_filled_turnover = 0.0

                # 重置周期内计数器
                period_orders_placed = 0
                period_orders_filled = 0
                period_buy_orders = 0
                period_sell_orders = 0
                period_buy_fills = 0
                period_sell_fills = 0
                period_long_close_profit = 0.0
                period_short_close_profit = 0.0

            # 更新当前bar的K线数据
            bar_high = max(bar_high, trade_price)
            bar_low = min(bar_low, trade_price)
            bar_close = trade_price
            bar_volume += trade_qty
            bar_turnover += trade_price * trade_qty

            # 统计买卖笔数
            if trade_side == 0:  # buy
                bar_buy_count += 1
            else:  # sell
                bar_sell_count += 1

            # 实时更新当前bar的最新状态（将在周期结束时保存）
            if bar_idx >= 0:
                trade_history[bar_idx][10] = cash
                trade_history[bar_idx][11] = position
                trade_history[bar_idx][12] = total_value
                trade_history[bar_idx][15] = current_pema
                trade_history[bar_idx][16] = current_ask_permit_val
                trade_history[bar_idx][17] = current_bid_permit_val

    # 收集未完成的存活订单到历史记录
    for i in range(max_live_orders):
        if live_order_indices[i] >= 0:
            order_idx = live_order_indices[i]
            for j in range(ORDER_FIELD_COUNT):
                order_history[order_history_count][j] = all_orders[order_idx][j]
            order_history[order_history_count][ORDER_FIELD_COUNT] = current_time
            order_history[order_history_count][ORDER_FIELD_COUNT + 1] = -1  # -1表示未完成
            order_history_count += 1

    # 处理最后一个bar（summary模式）
    if output_mode == 1 and current_period >= 0 and bar_idx >= 0:
        # 计算最后一个周期的时间戳
        period_start_ms = current_period * aggregation_seconds * 1000

        # 基础K线数据
        trade_history[bar_idx][0] = float64(period_start_ms)
        trade_history[bar_idx][1] = float64(period_start_ms)
        trade_history[bar_idx][2] = bar_open
        trade_history[bar_idx][3] = bar_high
        trade_history[bar_idx][4] = bar_low
        trade_history[bar_idx][5] = bar_close

        # 市场成交统计
        trade_history[bar_idx][6] = bar_volume
        trade_history[bar_idx][7] = bar_turnover
        trade_history[bar_idx][8] = float64(bar_buy_count)
        trade_history[bar_idx][9] = float64(bar_sell_count)

        # 账户状态和盈亏（已在实时更新中完成）
        bar_pnl = trade_history[bar_idx][12] - prev_total_value  # total_value - prev
        cumulative_pnl = trade_history[bar_idx][12] - initial_cash
        trade_history[bar_idx][13] = bar_pnl
        trade_history[bar_idx][14] = cumulative_pnl

        # 订单统计
        trade_history[bar_idx][18] = float64(period_orders_placed)
        trade_history[bar_idx][19] = float64(period_orders_filled)
        trade_history[bar_idx][20] = float64(period_buy_orders)
        trade_history[bar_idx][21] = float64(period_sell_orders)
        trade_history[bar_idx][22] = float64(period_buy_fills)
        trade_history[bar_idx][23] = float64(period_sell_fills)

        # 账户成交统计
        trade_history[bar_idx][24] = bar_filled_volume
        trade_history[bar_idx][25] = bar_filled_turnover

        # 平仓盈利
        trade_history[bar_idx][26] = period_long_close_profit
        trade_history[bar_idx][27] = period_short_close_profit
        bar_idx += 1

    # 裁剪trade_history到实际大小（summary模式）
    if output_mode == 1:
        trade_history = trade_history[:bar_idx]

    # 返回完整数据（包含持仓历史和逐笔trade历史）
    return (cash, position, fill_count, total_order_count, guard_fail_count,
            order_history[:order_history_count], position_history[:position_history_count], trade_history)


class BacktestEngine:
    """
    回测引擎包装类
    将numba加速的核心循环包装为易于使用的接口
    """

    def __init__(self,
                 initial_cash: float = 1000000.0,
                 initial_equity: float = 10000.0,
                 place_latency_config: dict = None,
                 cancel_latency_config: dict = None,
                 strategy_config: dict = None,
                 output_mode: str = 'detail',
                 aggregation_seconds: int = 1,
                 pos_punish: float = 0.0,
                 
                 ticksize: float = 0.1,
                 sigma_multi: float = 1.0,

                 use_dynamic_sigma_multi: bool = False,
                 use_dynamic_pos_punish: bool = False,
                 use_dynamic_price_tol: bool = False,
                 use_risk_off: bool=False):
        """
        初始化回测引擎

        参数：
            initial_cash: 初始资金
            initial_equity: 初始权益，用于计算固定下单量（每笔订单的目标成交金额）
            place_latency_config: 发单延时配置
            cancel_latency_config: 撤单延时配置
            strategy_config: 策略配置参数
            output_mode: 输出模式 'detail'(逐笔) 或 'summary'(聚合K线)
            aggregation_seconds: 聚合周期（秒），默认1秒，可设置600(10分钟)等
            pos_punish: 仓位惩罚因子(0.0~1.0)，0.0表示不惩罚，使用原始策略
        """
        self.initial_cash = initial_cash
        self.initial_equity = initial_equity
        self.aggregation_seconds = aggregation_seconds
        self.pos_punish = pos_punish

        # 延时配置
        place_config = place_latency_config or {'fixed': 10, 'random_min': 0, 'random_max': 5}
        cancel_config = cancel_latency_config or {'fixed': 10, 'random_min': 0, 'random_max': 5}

        self.place_fixed_latency = place_config.get('fixed', 10)
        self.place_random_min = place_config.get('random_min', 0)
        self.place_random_max = place_config.get('random_max', 5)
        self.cancel_fixed_latency = cancel_config.get('fixed', 10)
        self.cancel_random_min = cancel_config.get('random_min', 0)
        self.cancel_random_max = cancel_config.get('random_max', 5)

        # 策略配置
        strategy_config = strategy_config or {}
        self.price_tolerance = strategy_config.get('price_tolerance', 0.0002)  # 万分之二

        # 输出模式
        if output_mode not in ['detail', 'summary']:
            raise ValueError(f"无效的输出模式: {output_mode}. 必须是 'detail' 或 'summary'")
        self.output_mode = output_mode
        self.output_mode_int = 0 if output_mode == 'detail' else 1  # 转换为numba可用的整数

        # 回测结果
        self.final_cash = 0.0
        self.final_position = 0.0
        self.fill_count = 0
        self.total_orders = 0
        self.guard_fail_count = 0  # Guard失败统计
        self.order_history = None  # 完整订单历史数据
        self.position_history = None  # 持仓时间序列数据
        self.trade_history = None  # 逐笔trade完整状态数据


        self.ticksize = ticksize
        self.sigma_multi = sigma_multi
        self.use_dynamic_sigma_multi = use_dynamic_sigma_multi
        self.use_dynamic_pos_punish = use_dynamic_pos_punish
        self.use_dynamic_price_tol = use_dynamic_price_tol
        self.use_risk_off = use_risk_off

    def run_backtest(self, trades_df: pd.DataFrame,
                    progress_callback: Optional[Callable] = None) -> dict:
        """
        运行回测（使用numba加速的核心循环）

        参数：
            trades_df: trades数据DataFrame，必须包含：
                - timestamp: 时间戳（毫秒）
                - trade_price: 成交价格
                - trade_qty: 成交数量
                - trade_side: 成交方向（'buy' 或 'sell'）
                - guard_ask: Guard ask价位（可选）
                - guard_bid: Guard bid价位（可选）
                - ask_permit: 预计算的ask挂单价格（必需）
                - bid_permit: 预计算的bid挂单价格（必需）
            progress_callback: 进度回调函数（可选）

        返回：
            回测结果字典
        """
        print(f"🚀 开始numba加速回测，共 {len(trades_df)} 条trades")

        # 准备数据数组（numba需要numpy数组）
        # 时间戳字段适配
        if 'ts_exch' in trades_df.columns:
            trades_timestamp = trades_df['ts_exch'].values.astype(np.int64)
        elif 'timestamp' in trades_df.columns:
            trades_timestamp = trades_df['timestamp'].values.astype(np.int64)
        else:
            trades_timestamp = trades_df['transact_time'].values.astype(np.int64)

        # 价格字段适配
        if 'price' in trades_df.columns:
            trades_price = trades_df['price'].values.astype(np.float64)
        else:
            trades_price = trades_df['trade_price'].values.astype(np.float64)

        # 数量字段适配
        if 'qty' in trades_df.columns:
            trades_qty = trades_df['qty'].values.astype(np.float64)
        else:
            trades_qty = trades_df['trade_qty'].values.astype(np.float64)

        # 转换交易方向为数值
        if 'side' in trades_df.columns:
            trades_side_str = trades_df['side'].values
            trades_side = np.zeros(len(trades_side_str), dtype=np.int64)
            for i in range(len(trades_side_str)):
                trades_side[i] = 0 if trades_side_str[i] == 'buy' else 1
        elif 'trade_side' in trades_df.columns:
            trades_side_str = trades_df['trade_side'].values
            trades_side = np.zeros(len(trades_side_str), dtype=np.int64)
            for i in range(len(trades_side_str)):
                trades_side[i] = 0 if trades_side_str[i] == 'buy' else 1
        elif 'is_buyer_maker' in trades_df.columns:
            # 根据is_buyer_maker字段推断交易方向
            # is_buyer_maker=True表示买方是maker，即这笔交易是卖方主动成交，记为sell(1)
            # is_buyer_maker=False表示卖方是maker，即这笔交易是买方主动成交，记为buy(0)
            is_buyer_maker = trades_df['is_buyer_maker'].values
            trades_side = np.zeros(len(is_buyer_maker), dtype=np.int64)
            for i in range(len(is_buyer_maker)):
                trades_side[i] = 1 if is_buyer_maker[i] else 0  # True->1(sell), False->0(buy)
        else:
            raise ValueError("缺少交易方向字段：需要 'side'、'trade_side' 或 'is_buyer_maker' 字段")

        # 准备Guard数组（如果有）
        if 'guard_ask' in trades_df.columns and 'guard_bid' in trades_df.columns:
            guard_ask = trades_df['guard_ask'].values.astype(np.float64)
            guard_bid = trades_df['guard_bid'].values.astype(np.float64)
            print(f"✅ 使用Guard机制")
        else:
            # 如果没有Guard数据，使用默认值（不限制）
            guard_ask = np.full(len(trades_df), np.inf, dtype=np.float64)
            guard_bid = np.full(len(trades_df), -np.inf, dtype=np.float64)
            print(f"⚠️ 未发现Guard数据，不使用Guard机制")

        # 准备预计算指标数组（必需）
        if 'ask_permit' in trades_df.columns and 'bid_permit' in trades_df.columns:
            ask_permit = trades_df['ask_permit'].values.astype(np.float64)
            bid_permit = trades_df['bid_permit'].values.astype(np.float64)
            print(f"✅ 使用预计算挂单价格指标")
        else:
            raise ValueError("缺少必需的预计算指标：ask_permit 和 bid_permit")

        # 准备PEMA数组（必需）
        if 'pema' in trades_df.columns:
            pema = trades_df['pema'].values.astype(np.float64)
            print(f"✅ 使用PEMA公允价格指标")
        else:
            raise ValueError("缺少必需的预计算指标：pema")
        
        # sigma（如果不用动态 sigma_multi，可以给全 0）
        if 'sigma' in trades_df.columns:
            sigma = trades_df['sigma'].values.astype(np.float64)
        else:
            sigma = np.zeros(len(trades_df), dtype=np.float64)

        # sigma_multi_arr
        if 'sigma_multi_t' in trades_df.columns:
            sigma_multi_arr = trades_df['sigma_multi_t'].values.astype(np.float64)
        else:
            sigma_multi_arr = np.full(len(trades_df), 1.0, dtype=np.float64)  # 你也可以用常数默认

        # 动态 price tol
        if 'price_tol_t' in trades_df.columns:
            price_tol_arr = trades_df['price_tol_t'].values.astype(np.float64)
        else:
            price_tol_arr = np.full(len(trades_df), self.price_tolerance, dtype=np.float64)

        # 动态 pos punish
        if 'pos_punish_t' in trades_df.columns:
            pos_punish_arr = trades_df['pos_punish_t'].values.astype(np.float64)
        else:
            pos_punish_arr = np.full(len(trades_df), self.pos_punish, dtype=np.float64)

        # risk_off
        if 'risk_off' in trades_df.columns:
            risk_off_arr = trades_df['risk_off'].values.astype(np.bool_)
        else:
            risk_off_arr = np.zeros(len(trades_df), dtype=np.bool_)

        # 调用修正后的numba加速核心循环
        result = run_backtest_core(
            trades_timestamp, trades_price, trades_qty, trades_side,
            pema,
            sigma,
            guard_ask, guard_bid,
            ask_permit, bid_permit,

            self.initial_cash, self.initial_equity,

            self.price_tolerance,
            self.pos_punish,
            self.ticksize,

            sigma_multi_arr,
            price_tol_arr,
            pos_punish_arr,
            risk_off_arr,

            self.use_dynamic_sigma_multi,
            self.use_dynamic_pos_punish,
            self.use_dynamic_price_tol,
            self.use_risk_off,

            self.place_fixed_latency, self.place_random_min, self.place_random_max,
            self.cancel_fixed_latency, self.cancel_random_min, self.cancel_random_max,

            self.output_mode_int,
            self.aggregation_seconds
        )

        # 解析结果
        self.final_cash, self.final_position, self.fill_count, self.total_orders, self.guard_fail_count, self.order_history, self.position_history, self.trade_history = result

        print(f"✅ numba加速回测完成")

        # 保存最后的trade价格用于权益计算
        self.last_trade_price = trades_price[-1] if len(trades_price) > 0 else 0.0

        # 生成报告
        return self.generate_report()

    def save_csv_results(self, output_dir: str, symbol: str = "", params_str: str = "", date_str: str = ""):
        """
        保存回测结果到CSV文件（向后兼容）

        参数：
            output_dir: 基础输出目录
            symbol: 交易品种名
            params_str: 参数字符串（如 "tau_5min_sigma_1p0"）
            date_str: 日期字符串（如 "2024-07-08"）
        """
        import os
        import pandas as pd
        from pathlib import Path

        # 构建输出路径：币种/参数/日期/
        if symbol and params_str and date_str:
            final_output_dir = os.path.join(output_dir, symbol, params_str, date_str)
        else:
            final_output_dir = output_dir

        # 确保输出目录存在
        Path(final_output_dir).mkdir(parents=True, exist_ok=True)

        # 文件前缀
        file_prefix = f"{symbol}_" if symbol else ""

        # 初始化路径变量
        trade_details_path = None
        kline_path = None

        if self.output_mode == 'detail':
            # detail模式：保存逐笔交易数据
            trade_details_df = pd.DataFrame(self.trade_history, columns=[
                'trade_idx', 'timestamp', 'trade_price', 'trade_qty', 'trade_side',
                'pema', 'ask_permit', 'bid_permit', 'guard_ask', 'guard_bid',
                'cash', 'position', 'total_value', 'buy_orders_alive', 'sell_orders_alive', 'decision_flag'
            ])

            # 删除trade_idx和decision_flag列（不需要的内部字段）
            trade_details_df = trade_details_df.drop(['trade_idx', 'decision_flag'], axis=1)

            trade_details_path = os.path.join(final_output_dir, f"{file_prefix}trade_details.csv")
            trade_details_df.to_csv(trade_details_path, index=False)

        else:  # summary模式
            # 保存秒级K线数据
            kline_df = pd.DataFrame(self.trade_history, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'turnover', 'buy_count', 'sell_count',
                'cash', 'position', 'total_value',
                'buy_orders_alive', 'sell_orders_alive',
                'pema', 'ask_permit', 'bid_permit',
                'new_buy_orders', 'new_sell_orders', 'new_total_orders',
                'new_buy_fills', 'new_sell_fills', 'new_total_fills',
                'long_close_profit', 'short_close_profit'
            ])

            kline_path = os.path.join(final_output_dir, f"{file_prefix}kline_1s.csv")
            kline_df.to_csv(kline_path, index=False)

        # 2. 保存订单记录文件（两种模式都保存）
        order_records_df = pd.DataFrame(self.order_history, columns=[
            'order_id', 'price', 'quantity', 'side', 'status',
            'birth_time', 'arrive_time', 'cancel_time', 'ex_cancel_time',
            'fill_time', 'filled_qty', 'create_time', 'final_time'
        ])

        # 只保留需要的列并重新排序
        order_records_df = order_records_df[[
            'order_id', 'birth_time', 'arrive_time', 'price', 'quantity',
            'side', 'status', 'cancel_time', 'ex_cancel_time', 'fill_time', 'final_time'
        ]]

        # 重命名列为更直观的名称
        order_records_df.columns = [
            'order_id', 'create_time', 'arrive_time', 'price', 'quantity',
            'side', 'final_status', 'cancel_request_time', 'cancel_complete_time', 'fill_time', 'final_time'
        ]

        order_records_path = os.path.join(final_output_dir, f"{file_prefix}order_records.csv")
        order_records_df.to_csv(order_records_path, index=False)

        # 计算记录数
        if self.output_mode == 'detail':
            trade_count = len(trade_details_df)
            trade_data_path = trade_details_path
        else:
            trade_count = len(kline_df)
            trade_data_path = kline_path

        return {
            'output_dir': final_output_dir,
            'output_mode': self.output_mode,
            'trade_data_path': trade_data_path,
            'order_records_path': order_records_path,
            'trade_count': trade_count,
            'order_count': len(order_records_df)
        }

    def save_parquet_results(self, output_dir: str, symbol: str = "", params_str: str = "", date_str: str = ""):
        """
        保存回测结果到Parquet文件（高效压缩格式）

        路径格式：
        - detail模式：{output_dir}/{symbol}/{params_str}/{date_str}/
        - summary模式：{output_dir}/{symbol}/{params_str}/{date_str}/

        参数：
            output_dir: 基础输出目录
            symbol: 交易品种名
            params_str: 参数字符串（如 "tau_5min_sigma_1p0"）
            date_str: 日期字符串（如 "2024-07-08"）
        """
        import os
        import pandas as pd
        from pathlib import Path

        # 构建输出路径：币种/参数/日期/
        if symbol and params_str and date_str:
            final_output_dir = os.path.join(output_dir, symbol, params_str, date_str)
        else:
            final_output_dir = output_dir

        # 确保输出目录存在
        Path(final_output_dir).mkdir(parents=True, exist_ok=True)

        # 文件前缀
        file_prefix = f"{symbol}_" if symbol else ""

        # 初始化路径变量
        trade_details_path = None
        kline_path = None

        if self.output_mode == 'detail':
            # detail模式：保存逐笔交易数据
            trade_details_df = pd.DataFrame(self.trade_history, columns=[
                'trade_idx', 'timestamp', 'trade_price', 'trade_qty', 'trade_side',
                'pema', 'ask_permit', 'bid_permit', 'guard_ask', 'guard_bid',
                'cash', 'position', 'total_value', 'buy_orders_alive', 'sell_orders_alive', 'decision_flag'
            ])

            # 删除trade_idx和decision_flag列（不需要的内部字段）
            trade_details_df = trade_details_df.drop(['trade_idx', 'decision_flag'], axis=1)

            trade_details_path = os.path.join(final_output_dir, f"{file_prefix}trade_details.parquet")
            trade_details_df.to_parquet(trade_details_path, index=False, compression='snappy')

        else:  # summary模式
            # 保存聚合K线数据
            kline_df = pd.DataFrame(self.trade_history, columns=[
                'timestamp', 'datetime', 'open', 'high', 'low', 'close',
                'volume', 'turnover', 'buy_count', 'sell_count',
                'cash', 'position', 'total_value', 'bar_pnl', 'cumulative_pnl',
                'pema', 'ask_permit', 'bid_permit',
                'orders_placed', 'orders_filled', 'buy_orders', 'sell_orders',
                'buy_fills', 'sell_fills', 'filled_volume', 'filled_turnover',
                'long_close_profit', 'short_close_profit'
            ])

            # 转换datetime列
            kline_df['datetime'] = pd.to_datetime(kline_df['timestamp'], unit='ms')

            # 根据聚合周期生成文件名
            if self.aggregation_seconds == 1:
                filename = f"{file_prefix}kline_1s.parquet"
            elif self.aggregation_seconds == 60:
                filename = f"{file_prefix}kline_1min.parquet"
            elif self.aggregation_seconds == 300:
                filename = f"{file_prefix}kline_5min.parquet"
            elif self.aggregation_seconds == 600:
                filename = f"{file_prefix}kline_10min.parquet"
            elif self.aggregation_seconds == 3600:
                filename = f"{file_prefix}kline_1h.parquet"
            else:
                filename = f"{file_prefix}kline_{self.aggregation_seconds}s.parquet"

            kline_path = os.path.join(final_output_dir, filename)
            kline_df.to_parquet(kline_path, index=False, compression='snappy')

        # 2. 保存订单记录文件（两种模式都保存）
        order_records_df = pd.DataFrame(self.order_history, columns=[
            'order_id', 'price', 'quantity', 'side', 'status',
            'birth_time', 'arrive_time', 'cancel_time', 'ex_cancel_time',
            'fill_time', 'filled_qty', 'create_time', 'final_time'
        ])

        # 只保留需要的列并重新排序
        order_records_df = order_records_df[[
            'order_id', 'birth_time', 'arrive_time', 'price', 'quantity',
            'side', 'status', 'cancel_time', 'ex_cancel_time', 'fill_time', 'final_time'
        ]]

        # 重命名列为更直观的名称
        order_records_df.columns = [
            'order_id', 'create_time', 'arrive_time', 'price', 'quantity',
            'side', 'final_status', 'cancel_request_time', 'cancel_complete_time', 'fill_time', 'final_time'
        ]

        order_records_path = os.path.join(final_output_dir, f"{file_prefix}order_records.parquet")
        order_records_df.to_parquet(order_records_path, index=False, compression='snappy')

        # 计算记录数
        if self.output_mode == 'detail':
            trade_count = len(trade_details_df)
            trade_data_path = trade_details_path
        else:
            trade_count = len(kline_df)
            trade_data_path = kline_path

        return {
            'output_dir': final_output_dir,
            'output_mode': self.output_mode,
            'trade_data_path': trade_data_path,
            'order_records_path': order_records_path,
            'trade_count': trade_count,
            'order_count': len(order_records_df)
        }

    def generate_report(self) -> dict:
        """
        生成回测报告
        """
        # 使用实际的最后trade价格计算最终权益
        final_value = self.final_cash + self.final_position * self.last_trade_price

        report = {
            # 账户信息
            'initial_cash': self.initial_cash,
            'final_cash': self.final_cash,
            'final_position': self.final_position,
            'final_value': final_value,
            'last_trade_price': self.last_trade_price,  # 添加最后价格信息

            # 收益统计
            'total_return': (final_value - self.initial_cash) / self.initial_cash,

            # 交易统计
            'total_orders': self.total_orders,
            'filled_orders': self.fill_count,
            'fill_rate': self.fill_count / self.total_orders if self.total_orders > 0 else 0,

            # Guard统计
            'guard_fail_count': self.guard_fail_count,
            'guard_fail_rate': self.guard_fail_count / self.total_orders if self.total_orders > 0 else 0,

            # 成交历史
            'fill_count': self.fill_count,
        }

        # 添加订单历史详情
        if self.order_history is not None and len(self.order_history) > 0:
            # 提取成交订单信息
            filled_orders = self.order_history[self.order_history[:, 4] == ORDER_STATUS_FILLED]  # 状态列
            if len(filled_orders) > 0:
                order_ids = filled_orders[:, 0].astype(np.int64)  # 订单ID列
                prices = filled_orders[:, 1]  # 价格列
                report['fill_history_sample'] = {
                    'count': len(order_ids),
                    'first_10_order_ids': order_ids[:min(10, len(order_ids))].tolist(),
                    'price_range': (float(np.min(prices)), float(np.max(prices))),
                }

        return report

    def get_account_status(self) -> dict:
        """
        获取账户状态（兼容接口）
        """
        # 使用实际的最后trade价格
        last_price = getattr(self, 'last_trade_price', 0.0)
        return {
            'cash': self.final_cash,
            'position': self.final_position,
            'total_value': self.final_cash + self.final_position * last_price,
            'last_price': last_price,
        }

    def get_trade_history_df(self) -> pd.DataFrame:
        """
        获取逐笔trade历史数据的DataFrame格式

        返回：
            包含完整trade历史的DataFrame
        """
        if self.trade_history is None:
            return pd.DataFrame()

        # 创建DataFrame
        df = pd.DataFrame(self.trade_history, columns=[
            'trade_idx', 'timestamp', 'trade_price', 'trade_qty', 'trade_side',
            'pema', 'ask_permit', 'bid_permit', 'guard_ask', 'guard_bid',
            'cash', 'position', 'total_value',
            'buy_orders_alive', 'sell_orders_alive', 'decision_flag'
        ])

        # 转换数据类型
        df['trade_idx'] = df['trade_idx'].astype(np.int64)
        df['timestamp'] = df['timestamp'].astype(np.int64)
        df['trade_side'] = df['trade_side'].map({0: 'buy', 1: 'sell'})
        df['buy_orders_alive'] = df['buy_orders_alive'].astype(np.int32)
        df['sell_orders_alive'] = df['sell_orders_alive'].astype(np.int32)
        df['decision_flag'] = df['decision_flag'].astype(bool)

        return df


# 便捷函数
def run_simple_backtest(trades_df: pd.DataFrame,
                       initial_cash: float = 100000.0,
                       max_orders: int = 10000) -> dict:
    """
    便捷函数：运行简单回测

    参数：
        trades_df: trades数据DataFrame
        initial_cash: 初始资金
        max_orders: 最大订单数量

    返回：
        回测结果字典
    """
    engine = BacktestEngine(
        initial_cash=initial_cash,
        max_orders=max_orders
    )

    return engine.run_backtest(trades_df)


# ParameterOptimizer类已移除
# 新的策略系统使用预计算指标（ask_permit, bid_permit）进行智能做市
# 参数优化应该在指标计算阶段调整tau和sigma_multi参数