# -*- coding: utf-8 -*-
"""
基于Orderbook的做市策略 - 高层封装

提供易用的策略计算接口：
1. 管理参数配置
2. 数据预处理（trades、orderbook、PEMA）
3. 调用numba核心计算函数
4. 前向填充到trades时刻
5. 添加Guard价位
6. 返回与现有策略兼容的DataFrame格式
"""

import pandas as pd
import numpy as np
import time

from .indicators import compute_indicators_batch, parse_time_string
from .guard import compute_guard_batch
from .orderbook_indicators_optimized import compute_orderbook_strategy_core_optimized
from .orderbook_array_utils import prepare_orderbook_arrays_optimized
from ..utils.ticksize_manager import get_ticksize_manager


class OrderbookCalculator:
    """
    基于Orderbook的做市策略计算器

    策略思路：
    - 统计历史30分钟的成交流量分布
    - 计算每个档位的成交概率
    - 选择预期收益最大的档位挂单

    与现有框架的兼容性：
    - 输出DataFrame格式与动态波动率策略完全一致
    - 可直接使用BacktestEngine进行回测
    - 复用Guard机制、延时模型、撤单逻辑等
    """

    def __init__(self,
                 tau_p='5min',  # PEMA时间常数
                 window_seconds=1800,  # 流量统计窗口（秒），默认30分钟
                 bin_width=1000.0,  # 直方图箱体宽度（USDT）
                 initial_equity=10000.0,  # 单次建仓金额（USDT）
                 guard_k=100,  # Guard窗口大小
                 snapshot_interval=10,  # orderbook采样间隔（秒），固定为10
                 max_depth=100,  # orderbook最大深度
                 max_bins=1000,  # 直方图最大箱体数
                 time_unit=1000.0,  # 时间单位（毫秒）
                 eps=1e-12):
        """
        初始化计算器

        参数：
            tau_p: PEMA时间常数，支持字符串如'5min'、'1h'
            window_seconds: 流量统计窗口（秒），默认1800=30分钟
            bin_width: 直方图箱体宽度（USDT），默认1000
            initial_equity: 单次建仓金额（USDT），默认10000
            guard_k: Guard窗口大小，默认100
            snapshot_interval: orderbook采样间隔（秒），固定为10
            max_depth: orderbook最大深度，默认100
            max_bins: 直方图最大箱体数，默认1000
            time_unit: 时间单位（毫秒），默认1000.0
            eps: 数值计算精度，默认1e-12
        """
        self.tau_p = tau_p
        self.window_seconds = window_seconds
        self.bin_width = bin_width
        self.initial_equity = initial_equity
        self.guard_k = guard_k
        self.snapshot_interval = snapshot_interval
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.time_unit = time_unit
        self.eps = eps

        # 解析时间常数
        self.tau_p_seconds = parse_time_string(tau_p)

        print(f"[OrderbookCalculator] 初始化参数:")
        print(f"  tau_p = {tau_p} ({self.tau_p_seconds:.0f}秒)")
        print(f"  window_seconds = {window_seconds}秒 (流量统计窗口)")
        print(f"  bin_width = {bin_width} USDT")
        print(f"  initial_equity = {initial_equity} USDT")
        print(f"  guard_k = {guard_k}")

    def calculate(self, trades_df: pd.DataFrame, orderbook_df: pd.DataFrame,
                  symbol: str) -> pd.DataFrame:
        """
        主入口：计算基于orderbook的最优挂单价格

        参数：
            trades_df: Trades DataFrame，必须包含字段：
                - timestamp: int64，毫秒时间戳
                - trade_price: float64
                - trade_qty: float64
                - is_buyer_maker: bool
            orderbook_df: Orderbook DataFrame，必须包含字段：
                - timestamp: int64，毫秒时间戳
                - side: str，'bid' or 'ask'
                - price: float64
                - amount: float64
            symbol: 交易对符号，用于获取ticksize

        返回：
            enhanced_df: DataFrame，包含以下字段：
                - timestamp: int64，毫秒时间戳
                - trade_price: float64
                - trade_qty: float64
                - is_buyer_maker: bool
                - pema: float64，公允价格
                - ask_permit: float64，建议ask价格
                - bid_permit: float64，建议bid价格
                - guard_ask: float64，Guard ask价位
                - guard_bid: float64，Guard bid价位
                - ask_prob: float64，ask成交概率
                - bid_prob: float64，bid成交概率
                - buy_flow: float64，买方流量（USDT/截面）
                - sell_flow: float64，卖方流量（USDT/截面）
                - ask_profit: float64，ask预期收益（USDT）
                - bid_profit: float64，bid预期收益（USDT）
                - ask_cumulative: float64，ask累计挂单金额（USDT）
                - bid_cumulative: float64，bid累计挂单金额（USDT）
                - ask_order_size: float64，ask挂单量（币）
                - bid_order_size: float64，bid挂单量（币）
                - ob_best_ask: float64，orderbook最优ask价格
                - ob_best_bid: float64，orderbook最优bid价格
                - ob_best_ask_amount: float64，orderbook最优ask数量
                - ob_best_bid_amount: float64，orderbook最优bid数量

        流程：
            1. 预处理：计算PEMA（复用现有indicators模块）
            2. 预处理：将orderbook转换为numpy数组
            3. 调用核心计算函数（numba加速）
            4. 前向填充到trades时刻
            5. 添加Guard价位
            6. 返回完整DataFrame
        """
        print(f"\n[OrderbookCalculator] 开始计算 {symbol}")
        start_time = time.time()

        # 验证输入数据
        self._validate_trades_df(trades_df)
        self._validate_orderbook_df(orderbook_df)

        # 1. 计算PEMA（复用现有指标计算）
        print(f"[OrderbookCalculator] 步骤1: 计算PEMA...")
        pema_df = self._compute_pema(trades_df)

        # 2. 将orderbook转换为numpy数组
        print(f"[OrderbookCalculator] 步骤2: 预处理orderbook数据...")
        ob_arrays = self._prepare_orderbook_arrays(orderbook_df)

        # 3. 调用核心计算函数
        print(f"[OrderbookCalculator] 步骤3: 计算最优档位（numba加速）...")
        result_arrays = self._call_core_function(trades_df, ob_arrays, pema_df)

        # 4. 转换为DataFrame
        print(f"[OrderbookCalculator] 步骤4: 构造结果DataFrame...")
        result_df = self._arrays_to_dataframe(result_arrays)

        # 5. 前向填充到trades时刻
        print(f"[OrderbookCalculator] 步骤5: 前向填充到trades时刻...")
        filled_df = self._interpolate_to_trades(trades_df, result_df)

        # 6. 添加Guard价位
        print(f"[OrderbookCalculator] 步骤6: 添加Guard价位...")
        final_df = self._add_guard_prices(filled_df, symbol)

        elapsed = time.time() - start_time
        print(f"[OrderbookCalculator] 完成！耗时 {elapsed:.2f}秒")
        print(f"[OrderbookCalculator] 输出行数: {len(final_df):,}")

        return final_df

    def _validate_trades_df(self, df: pd.DataFrame):
        """验证trades DataFrame格式"""
        required_cols = ['timestamp', 'trade_price', 'trade_qty', 'is_buyer_maker']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Trades DataFrame缺少字段: {missing}")

        if df['timestamp'].dtype != np.int64:
            raise ValueError(f"timestamp字段类型错误: {df['timestamp'].dtype}")

    def _validate_orderbook_df(self, df: pd.DataFrame):
        """验证orderbook DataFrame格式"""
        required_cols = ['timestamp', 'side', 'price', 'amount']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Orderbook DataFrame缺少字段: {missing}")

        if df['timestamp'].dtype != np.int64:
            raise ValueError(f"timestamp字段类型错误: {df['timestamp'].dtype}")

    def _compute_pema(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算PEMA（复用现有indicators模块）

        返回：
            DataFrame，包含字段：
                - timestamp: int64
                - pema: float64
        """
        # 准备输入数据
        timestamps = trades_df['timestamp'].values
        prices = trades_df['trade_price'].values
        volumes = trades_df['trade_qty'].values  # 成交量

        # 调用现有的指标计算函数（只计算PEMA，不计算波动率）
        # 注意：这里我们只需要PEMA，所以固定sigma_multi=1.0, min_volatility=0
        # 返回10个值：vema, pema_prior, pema, vsqema, vema_o, msq_per_trade, sigma, fair, ask_permit, bid_permit
        _, _, pema_vals, _, _, _, _, _, _, _ = compute_indicators_batch(
            timestamps=timestamps,
            prices=prices,
            volumes=volumes,
            tau_p=self.tau_p_seconds,
            tau_o=self.tau_p_seconds,  # 不影响PEMA计算
            sigma_multi=1.0,
            min_volatility=0.0,
            ticksize=1e-10,  # 不影响PEMA计算
            time_unit=self.time_unit,
            eps=self.eps,
            dt_min=1e-3
        )

        return pd.DataFrame({
            'timestamp': timestamps,
            'pema': pema_vals
        })

    def _prepare_orderbook_arrays(self, orderbook_df: pd.DataFrame) -> dict:
        """
        将orderbook DataFrame转换为numba兼容的numpy数组

        使用numba优化版本，相比原pandas版本提速约5倍

        返回：
            dict，包含：
                - ob_timestamps: 1D数组，shape=(n_snapshots,)
                - ob_ask_prices: 2D数组，shape=(n_snapshots, max_depth)
                - ob_ask_amounts: 2D数组，shape=(n_snapshots, max_depth)
                - ob_bid_prices: 2D数组，shape=(n_snapshots, max_depth)
                - ob_bid_amounts: 2D数组，shape=(n_snapshots, max_depth)
                - ob_depths: 1D数组，shape=(n_snapshots,)，每个截面的实际档位数
        """
        # 使用优化版本（numba加速）
        return prepare_orderbook_arrays_optimized(
            orderbook_df,
            max_depth=self.max_depth
        )

    def _call_core_function(self, trades_df: pd.DataFrame,
                           ob_arrays: dict, pema_df: pd.DataFrame) -> tuple:
        """
        调用numba核心计算函数

        返回：
            (result_timestamps, result_pema, result_ask_permits, result_bid_permits,
             result_ask_probs, result_bid_probs)
        """
        # 准备trades数据
        trade_times = trades_df['timestamp'].values
        trade_prices = trades_df['trade_price'].values
        trade_qtys = trades_df['trade_qty'].values
        trade_is_buyer_maker = trades_df['is_buyer_maker'].values

        # 准备PEMA数据
        pema_times = pema_df['timestamp'].values
        pema_vals = pema_df['pema'].values

        # 调用核心函数（使用优化版本）
        return compute_orderbook_strategy_core_optimized(
            # Trades数据
            trade_times,
            trade_prices,
            trade_qtys,
            trade_is_buyer_maker,
            # Orderbook数据
            ob_arrays['ob_timestamps'],
            ob_arrays['ob_ask_prices'],
            ob_arrays['ob_ask_amounts'],
            ob_arrays['ob_bid_prices'],
            ob_arrays['ob_bid_amounts'],
            ob_arrays['ob_depths'],
            # PEMA数据
            pema_times,
            pema_vals,
            # 参数
            self.window_seconds,
            self.bin_width,
            self.initial_equity,
            self.snapshot_interval,
            self.max_bins
        )

    def _arrays_to_dataframe(self, result_arrays: tuple) -> pd.DataFrame:
        """将结果数组转换为DataFrame"""
        (result_timestamps, result_pema,
         result_ask_permits, result_bid_permits,
         result_ask_probs, result_bid_probs,
         result_buy_flow, result_sell_flow,
         result_ask_profit, result_bid_profit,
         result_ask_cumulative, result_bid_cumulative,
         result_ask_order_size, result_bid_order_size,
         result_ob_best_ask, result_ob_best_bid,
         result_ob_best_ask_amount, result_ob_best_bid_amount) = result_arrays

        return pd.DataFrame({
            'timestamp': result_timestamps,
            'pema': result_pema,
            'ask_permit': result_ask_permits,
            'bid_permit': result_bid_permits,
            'ask_prob': result_ask_probs,
            'bid_prob': result_bid_probs,
            # 新增字段
            'buy_flow': result_buy_flow,
            'sell_flow': result_sell_flow,
            'ask_profit': result_ask_profit,
            'bid_profit': result_bid_profit,
            'ask_cumulative': result_ask_cumulative,
            'bid_cumulative': result_bid_cumulative,
            'ask_order_size': result_ask_order_size,
            'bid_order_size': result_bid_order_size,
            'ob_best_ask': result_ob_best_ask,
            'ob_best_bid': result_ob_best_bid,
            'ob_best_ask_amount': result_ob_best_ask_amount,
            'ob_best_bid_amount': result_ob_best_bid_amount
        })

    def _interpolate_to_trades(self, trades_df: pd.DataFrame,
                               result_df: pd.DataFrame) -> pd.DataFrame:
        """
        前向填充：将10秒截面的结果填充到每个trade时刻

        使用merge_asof with direction='backward'确保不泄露未来数据

        返回：
            完整DataFrame，每个trade都有对应的指标值
        """
        # 只保留trades的基础字段
        base_df = trades_df[['timestamp', 'trade_price', 'trade_qty', 'is_buyer_maker']].copy()

        # 前向填充（只使用过去的数据）
        filled_df = pd.merge_asof(
            base_df,
            result_df,
            on='timestamp',
            direction='backward'  # 只使用<=当前时间的数据
        )

        return filled_df

    def _add_guard_prices(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加Guard价位（复用现有guard模块）

        返回：
            添加了guard_ask和guard_bid字段的DataFrame
        """
        # 获取ticksize
        ticksize_mgr = get_ticksize_manager()
        ticksize = ticksize_mgr.get_ticksize(symbol)

        # 如果ticksize为None，根据价格范围估算一个合理的默认值
        if ticksize is None:
            # 使用价格中位数的0.01%作为默认ticksize
            median_price = df['trade_price'].median()
            ticksize = median_price * 0.0001
            print(f"[OrderbookCalculator] 警告: {symbol} ticksize未配置，使用默认值 {ticksize:.8f}")

        # 计算Guard价位
        guard_ask, guard_bid = compute_guard_batch(
            timestamps=df['timestamp'].values,
            prices=df['trade_price'].values,
            is_buyer_maker=df['is_buyer_maker'].values,
            k_window=self.guard_k,
            ticksize=ticksize
        )

        # 添加到DataFrame
        df['guard_ask'] = guard_ask
        df['guard_bid'] = guard_bid

        return df
