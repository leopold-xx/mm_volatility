# -*- coding: utf-8 -*-
"""
基于多维数组的订单管理系统
完全numba兼容，用于极致性能的回测
"""

import numpy as np
from numba import jit, int64, float64, boolean
from enum import IntEnum

# 订单状态常量
ORDER_STATUS_PENDING = 0      # 待到达：订单已创建但未到达交易所
ORDER_STATUS_ACTIVE = 1       # 活跃：订单已到达交易所，可参与撮合
ORDER_STATUS_CANCELLING = 2   # 撤单中：收到撤单指令，等待撤单生效
ORDER_STATUS_FILLED = 3       # 成交：订单已完全成交
ORDER_STATUS_CANCELLED = 4    # 已撤销：撤单生效，订单被撤销
ORDER_STATUS_FAILED = 5       # 失败：违反Guard条件，订单被拒绝

# 订单方向常量
ORDER_SIDE_BUY = 0           # 买单
ORDER_SIDE_SELL = 1          # 卖单

# 订单数组字段位置常量（重要：约定各字段在数组中的位置）
ORDER_ID_IDX = 0             # 订单ID
ORDER_PRICE_IDX = 1          # 订单价格
ORDER_QUANTITY_IDX = 2       # 订单数量
ORDER_SIDE_IDX = 3           # 订单方向（0=买，1=卖）
ORDER_STATUS_IDX = 4         # 订单状态
ORDER_BIRTH_TIME_IDX = 5     # 订单创建时间
ORDER_ARRIVE_TIME_IDX = 6    # 到达交易所时间
ORDER_CANCEL_TIME_IDX = 7    # 撤单请求时间（-1表示未撤单）
ORDER_EX_CANCEL_TIME_IDX = 8 # 撤单生效时间（-1表示未撤单）
ORDER_FILL_TIME_IDX = 9      # 成交时间（-1表示未成交）
ORDER_FILLED_QTY_IDX = 10    # 已成交数量

# 订单数组字段总数
ORDER_FIELD_COUNT = 11


@jit(nopython=True)
def calculate_latency(fixed_latency: int64,
                     random_min: int64,
                     random_max: int64) -> int64:
    """
    计算延时（固定延时 + 随机延时）
    """
    total_latency = fixed_latency
    if random_min < random_max:
        random_latency = np.random.randint(random_min, random_max + 1)
        total_latency += random_latency
    return total_latency


@jit(nopython=True)
def create_order_array(order_id: int64,
                      price: float64,
                      quantity: float64,
                      side: int64,
                      birth_time: int64,
                      place_fixed_latency: int64,
                      place_random_min: int64,
                      place_random_max: int64) -> np.ndarray:
    """
    创建订单数组（numba加速版本）

    返回：
        长度为ORDER_FIELD_COUNT的一维数组，包含订单所有信息
    """
    order = np.zeros(ORDER_FIELD_COUNT, dtype=np.float64)

    # 计算到达时间
    place_latency = calculate_latency(place_fixed_latency, place_random_min, place_random_max)
    arrive_time = birth_time + place_latency

    # 设置订单字段
    order[ORDER_ID_IDX] = float64(order_id)
    order[ORDER_PRICE_IDX] = price
    order[ORDER_QUANTITY_IDX] = quantity
    order[ORDER_SIDE_IDX] = float64(side)
    order[ORDER_STATUS_IDX] = float64(ORDER_STATUS_PENDING)
    order[ORDER_BIRTH_TIME_IDX] = float64(birth_time)
    order[ORDER_ARRIVE_TIME_IDX] = float64(arrive_time)
    order[ORDER_CANCEL_TIME_IDX] = -1.0     # -1表示未撤单
    order[ORDER_EX_CANCEL_TIME_IDX] = -1.0  # -1表示未撤单
    order[ORDER_FILL_TIME_IDX] = -1.0       # -1表示未成交
    order[ORDER_FILLED_QTY_IDX] = 0.0       # 初始成交量为0

    return order


@jit(nopython=True)
def update_order_status_by_time(order: np.ndarray, current_time: int64) -> boolean:
    """
    根据当前时间更新订单状态

    参数：
        order: 订单数组
        current_time: 当前时间戳

    返回：
        状态是否发生变化
    """
    old_status = int64(order[ORDER_STATUS_IDX])

    # 检查撤单生效（优先级最高）
    if (order[ORDER_EX_CANCEL_TIME_IDX] != -1 and
        current_time >= order[ORDER_EX_CANCEL_TIME_IDX] and
        old_status != ORDER_STATUS_FILLED):  # 已成交的订单不能再撤销
        order[ORDER_STATUS_IDX] = float64(ORDER_STATUS_CANCELLED)

    # 检查撤单开始
    elif (order[ORDER_CANCEL_TIME_IDX] != -1 and
          current_time >= order[ORDER_CANCEL_TIME_IDX] and
          old_status == ORDER_STATUS_ACTIVE):
        order[ORDER_STATUS_IDX] = float64(ORDER_STATUS_CANCELLING)

    # 检查订单到达
    elif (current_time >= order[ORDER_ARRIVE_TIME_IDX] and
          old_status == ORDER_STATUS_PENDING):
        order[ORDER_STATUS_IDX] = float64(ORDER_STATUS_ACTIVE)

    new_status = int64(order[ORDER_STATUS_IDX])
    return old_status != new_status


@jit(nopython=True)
def initiate_order_cancel(order: np.ndarray,
                         cancel_time: int64,
                         cancel_fixed_latency: int64,
                         cancel_random_min: int64,
                         cancel_random_max: int64) -> boolean:
    """
    发起订单撤销（设置撤单时间戳）
    """
    # 只能撤销活跃状态的订单
    if int64(order[ORDER_STATUS_IDX]) != ORDER_STATUS_ACTIVE:
        return False

    # 计算撤单生效时间
    cancel_latency = calculate_latency(cancel_fixed_latency, cancel_random_min, cancel_random_max)
    ex_cancel_time = cancel_time + cancel_latency

    # 设置撤单时间戳
    order[ORDER_CANCEL_TIME_IDX] = float64(cancel_time)
    order[ORDER_EX_CANCEL_TIME_IDX] = float64(ex_cancel_time)

    return True


@jit(nopython=True)
def fill_order(order: np.ndarray, fill_time: int64) -> boolean:
    """
    成交订单
    """
    # 只有活跃状态的订单才能成交
    if int64(order[ORDER_STATUS_IDX]) != ORDER_STATUS_ACTIVE:
        return False

    # 设置成交信息
    order[ORDER_FILL_TIME_IDX] = float64(fill_time)
    order[ORDER_FILLED_QTY_IDX] = order[ORDER_QUANTITY_IDX]  # 全部成交
    order[ORDER_STATUS_IDX] = float64(ORDER_STATUS_FILLED)

    return True


@jit(nopython=True)
def check_fill_condition(order: np.ndarray, trade_price: float64) -> boolean:
    """
    检查订单是否满足成交条件（最悲观估计）

    买单成交条件：trade价格严格小于订单价格
    卖单成交条件：trade价格严格大于订单价格
    """
    # 只有活跃状态的订单才可能成交
    if int64(order[ORDER_STATUS_IDX]) != ORDER_STATUS_ACTIVE:
        return False

    order_price = order[ORDER_PRICE_IDX]
    order_side = int64(order[ORDER_SIDE_IDX])

    # 根据订单方向检查成交条件
    if order_side == ORDER_SIDE_BUY:
        return trade_price < order_price  # 最悲观：严格小于
    else:  # SELL
        return trade_price > order_price  # 最悲观：严格大于


@jit(nopython=True)
def batch_update_orders_status(orders: np.ndarray,
                              order_count: int64,
                              current_time: int64) -> int64:
    """
    批量更新订单状态（numba加速核心函数）

    参数：
        orders: 订单二维数组 [max_orders, ORDER_FIELD_COUNT]
        order_count: 有效订单数量
        current_time: 当前时间戳

    返回：
        状态发生变化的订单数量
    """
    changed_count = 0

    for i in range(order_count):
        if update_order_status_by_time(orders[i], current_time):
            changed_count += 1

    return changed_count


@jit(nopython=True)
def batch_check_order_fills(orders: np.ndarray,
                           order_count: int64,
                           trade_price: float64,
                           fill_time: int64) -> np.ndarray:
    """
    批量检查订单成交条件（numba加速核心函数）

    参数：
        orders: 订单二维数组
        order_count: 有效订单数量
        trade_price: 成交价格
        fill_time: 成交时间戳

    返回：
        成交订单的索引数组
    """
    # 预分配足够大的数组存储成交订单索引
    filled_indices = np.zeros(order_count, dtype=np.int64)
    filled_count = 0

    for i in range(order_count):
        order = orders[i]

        # 检查成交条件
        if check_fill_condition(order, trade_price):
            # 执行成交
            if fill_order(order, fill_time):
                filled_indices[filled_count] = i
                filled_count += 1

    # 返回实际成交的索引数组
    return filled_indices[:filled_count]


@jit(nopython=True)
def get_orders_by_status(orders: np.ndarray,
                        order_count: int64,
                        target_status: int64) -> np.ndarray:
    """
    获取指定状态的订单索引

    返回：
        符合条件的订单索引数组
    """
    matching_indices = np.zeros(order_count, dtype=np.int64)
    match_count = 0

    for i in range(order_count):
        if int64(orders[i][ORDER_STATUS_IDX]) == target_status:
            matching_indices[match_count] = i
            match_count += 1

    return matching_indices[:match_count]


@jit(nopython=True)
def find_order_by_id(orders: np.ndarray,
                    order_count: int64,
                    target_order_id: int64) -> int64:
    """
    根据订单ID查找订单索引

    返回：
        订单索引，未找到返回-1
    """
    for i in range(order_count):
        if int64(orders[i][ORDER_ID_IDX]) == target_order_id:
            return i
    return -1


@jit(nopython=True)
def find_active_orders_by_side(orders: np.ndarray,
                              order_count: int64,
                              side: int64) -> np.ndarray:
    """
    查找指定方向的活跃订单（ACTIVE状态）

    参数：
        orders: 订单数组
        order_count: 订单总数
        side: 订单方向（0=买，1=卖）

    返回：
        活跃订单的索引数组
    """
    active_indices = []
    for i in range(order_count):
        if (int64(orders[i][ORDER_STATUS_IDX]) == ORDER_STATUS_ACTIVE and
            int64(orders[i][ORDER_SIDE_IDX]) == side):
            active_indices.append(i)

    return np.array(active_indices, dtype=np.int64)


@jit(nopython=True)
def find_pending_orders_by_side(orders: np.ndarray,
                               order_count: int64,
                               side: int64) -> np.ndarray:
    """
    查找指定方向的待处理订单（PENDING状态）
    """
    pending_indices = []
    for i in range(order_count):
        if (int64(orders[i][ORDER_STATUS_IDX]) == ORDER_STATUS_PENDING and
            int64(orders[i][ORDER_SIDE_IDX]) == side):
            pending_indices.append(i)

    return np.array(pending_indices, dtype=np.int64)


@jit(nopython=True)
def find_cancelling_orders_by_side(orders: np.ndarray,
                                  order_count: int64,
                                  side: int64) -> np.ndarray:
    """
    查找指定方向的撤单中订单（CANCELLING状态）
    """
    cancelling_indices = []
    for i in range(order_count):
        if (int64(orders[i][ORDER_STATUS_IDX]) == ORDER_STATUS_CANCELLING and
            int64(orders[i][ORDER_SIDE_IDX]) == side):
            cancelling_indices.append(i)

    return np.array(cancelling_indices, dtype=np.int64)


@jit(nopython=True)
def has_living_orders_by_side(orders: np.ndarray,
                             order_count: int64,
                             side: int64) -> boolean:
    """
    检查指定方向是否有存续期订单（PENDING/ACTIVE/CANCELLING状态）
    """
    for i in range(order_count):
        if int64(orders[i][ORDER_SIDE_IDX]) == side:
            status = int64(orders[i][ORDER_STATUS_IDX])
            if (status == ORDER_STATUS_PENDING or
                status == ORDER_STATUS_ACTIVE or
                status == ORDER_STATUS_CANCELLING):
                return True
    return False


@jit(nopython=True)
def calculate_price_deviation(price1: float64, price2: float64) -> float64:
    """
    计算两个价格的相对偏差（绝对值）
    """
    if price1 <= 0.0 or price2 <= 0.0:
        return float64(1.0)  # 返回很大的偏差，表示需要改单

    return abs(price1 - price2) / max(price1, price2)


@jit(nopython=True)
def check_guard_fail(order: np.ndarray,
                    guard_ask: float64,
                    guard_bid: float64) -> boolean:
    """
    检查订单是否违反Guard条件

    规则：
    - 买单价格 >= guard_ask → fail
    - 卖单价格 <= guard_bid → fail

    参数：
        order: 订单数组
        guard_ask: 当前guard_ask
        guard_bid: 当前guard_bid

    返回：
        是否fail
    """
    order_price = order[ORDER_PRICE_IDX]
    order_side = int64(order[ORDER_SIDE_IDX])

    if order_side == ORDER_SIDE_BUY:
        # 买单价格必须严格小于guard_ask
        return order_price >= guard_ask
    else:  # SELL
        # 卖单价格必须严格大于guard_bid
        return order_price <= guard_bid


@jit(nopython=True)
def update_position_and_cash(cash: float64,
                            position: float64,
                            order: np.ndarray) -> tuple:
    """
    根据成交订单更新持仓和资金

    返回：
        (新资金, 新持仓)
    """
    order_price = order[ORDER_PRICE_IDX]
    order_qty = order[ORDER_QUANTITY_IDX]
    order_side = int64(order[ORDER_SIDE_IDX])

    if order_side == ORDER_SIDE_BUY:
        # 买单成交：减少现金，增加持仓
        cost = order_price * order_qty
        new_cash = cash - cost
        new_position = position + order_qty
    else:  # SELL
        # 卖单成交：增加现金，减少持仓
        revenue = order_price * order_qty
        new_cash = cash + revenue
        new_position = position - order_qty

    return new_cash, new_position