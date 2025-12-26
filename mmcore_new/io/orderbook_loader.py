# -*- coding: utf-8 -*-
"""
Orderbook截面数据加载器

功能：
1. 加载单日orderbook截面parquet文件
2. 过滤无效数据（price=-1, amount=0）
3. 时间戳单位转换（微秒→毫秒，统一格式）
4. 数据验证和完整性检查
5. 返回标准DataFrame格式
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class OrderbookLoader:
    """
    Orderbook截面数据加载器

    数据格式：
    - 文件命名: orderbook_snapshot_{symbol}_{date}_{interval}.parquet
    - 采样间隔: 10秒
    - 时间戳单位: 微秒（UTC）
    - 字段: timestamp(int64), side(str), price(float64), amount(float64)
    """

    def __init__(self, data_dir: str = 'sample_data/orderbook_snapshots'):
        """
        初始化加载器

        参数：
            data_dir: orderbook数据目录路径
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Orderbook数据目录不存在: {self.data_dir}")

    def load(self, symbol: str, date: str,
             filter_invalid: bool = True,
             convert_to_ms: bool = True,
             snapshot_suffix: str = '10s') -> pd.DataFrame:
        """
        加载单日orderbook截面数据

        参数：
            symbol: 交易对符号，如'ASTERUSDT'
            date: 日期字符串，格式'YYYY-MM-DD'，如'2025-09-19'
            filter_invalid: 是否过滤无效数据（price=-1, amount=0）
            convert_to_ms: 是否转换时间戳为毫秒（默认True）
            snapshot_suffix: 文件名后缀，如'10s'或'500ms'（默认'10s'）

        返回：
            DataFrame，包含字段：
                timestamp: int64，毫秒时间戳（如果convert_to_ms=True）
                side: str，'bid' or 'ask'
                price: float64，档位价格
                amount: float64，档位数量
        """
        # 构造文件路径
        filename = f'orderbook_snapshot_{symbol}_{date}_{snapshot_suffix}.parquet'
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Orderbook文件不存在: {filepath}")

        # 读取parquet文件
        df = pd.read_parquet(filepath)

        # 验证字段完整性
        required_columns = ['timestamp', 'side', 'price', 'amount']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"缺少必要字段: {missing_columns}")

        # 验证数据类型
        if df['timestamp'].dtype != np.int64:
            raise ValueError(f"timestamp字段类型错误: {df['timestamp'].dtype}，期望int64")
        if df['price'].dtype != np.float64:
            raise ValueError(f"price字段类型错误: {df['price'].dtype}，期望float64")
        if df['amount'].dtype != np.float64:
            raise ValueError(f"amount字段类型错误: {df['amount'].dtype}，期望float64")

        # 过滤无效数据
        if filter_invalid:
            original_len = len(df)
            df = df[(df['price'] > 0) & (df['amount'] > 0)].copy()
            filtered_count = original_len - len(df)
            if filtered_count > 0:
                print(f"[OrderbookLoader] 过滤了 {filtered_count} 行无效数据 "
                      f"({filtered_count/original_len*100:.1f}%)")

        # 转换时间戳单位：微秒 → 毫秒
        if convert_to_ms:
            df['timestamp'] = (df['timestamp'] / 1000).astype(np.int64)

        # 验证side字段值
        valid_sides = {'bid', 'ask'}
        invalid_sides = set(df['side'].unique()) - valid_sides
        if invalid_sides:
            raise ValueError(f"side字段包含无效值: {invalid_sides}，期望'bid'或'ask'")

        # 检查时间连续性
        # 根据snapshot_suffix推断期望的时间间隔
        if 'ms' in snapshot_suffix:
            # 如500ms
            expected_interval_ms = int(snapshot_suffix.replace('ms', ''))
        elif 's' in snapshot_suffix:
            # 如10s
            expected_interval_ms = int(snapshot_suffix.replace('s', '')) * 1000
        else:
            expected_interval_ms = 10000  # 默认10秒

        self._validate_time_continuity(df, symbol, date, expected_interval_ms)

        # 重置索引
        df = df.reset_index(drop=True)

        print(f"[OrderbookLoader] 成功加载 {symbol} {date} 数据: "
              f"{len(df):,}行, {len(df['timestamp'].unique())}个截面")

        return df

    def _validate_time_continuity(self, df: pd.DataFrame, symbol: str, date: str, expected_interval_ms: int = 10000):
        """
        验证时间连续性

        检查：
        1. 时间戳是否升序
        2. 时间间隔是否均匀
        3. 是否有缺失截面

        参数：
            expected_interval_ms: 期望的时间间隔（毫秒），默认10000（10秒）
        """
        timestamps = sorted(df['timestamp'].unique())

        if len(timestamps) == 0:
            raise ValueError(f"没有有效的时间截面")

        # 检查时间间隔
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            min_interval = intervals.min()
            max_interval = intervals.max()

            # 允许±10%的误差
            lower_bound = expected_interval_ms * 0.9
            upper_bound = expected_interval_ms * 1.1

            # 检查是否接近期望间隔
            if not (lower_bound <= min_interval <= upper_bound and lower_bound <= max_interval <= upper_bound):
                print(f"[OrderbookLoader] 警告: 时间间隔不均匀 - "
                      f"最小={min_interval}ms, 最大={max_interval}ms, 期望={expected_interval_ms}ms")

        # 检查有效截面比例
        total_snapshots = len(timestamps)
        valid_snapshots = 0
        for ts in timestamps:
            snapshot = df[df['timestamp'] == ts]
            has_bid = (snapshot['side'] == 'bid').any()
            has_ask = (snapshot['side'] == 'ask').any()
            if has_bid and has_ask:
                valid_snapshots += 1

        valid_ratio = valid_snapshots / total_snapshots if total_snapshots > 0 else 0
        print(f"[OrderbookLoader] 有效截面（bid和ask都有数据）: "
              f"{valid_snapshots}/{total_snapshots} ({valid_ratio*100:.1f}%)")

        if valid_ratio < 0.1:
            print(f"[OrderbookLoader] 警告: 有效截面比例过低 ({valid_ratio*100:.1f}%)，"
                  f"可能影响回测结果")

    def load_multiple_days(self, symbol: str, date_list: list,
                          filter_invalid: bool = True,
                          convert_to_ms: bool = True,
                          snapshot_suffix: str = '10s') -> pd.DataFrame:
        """
        加载多日orderbook数据并合并

        参数：
            symbol: 交易对符号
            date_list: 日期列表，如['2025-09-19', '2025-09-20']
            filter_invalid: 是否过滤无效数据
            convert_to_ms: 是否转换时间戳为毫秒
            snapshot_suffix: 文件名后缀，如'10s'或'500ms'（默认'10s'）

        返回：
            合并后的DataFrame，按时间戳升序排列
        """
        dfs = []
        for date in date_list:
            try:
                df = self.load(symbol, date, filter_invalid, convert_to_ms, snapshot_suffix)
                dfs.append(df)
            except FileNotFoundError as e:
                print(f"[OrderbookLoader] 跳过日期 {date}: {e}")
                continue

        if not dfs:
            raise ValueError(f"没有成功加载任何日期的数据")

        # 合并所有数据
        combined_df = pd.concat(dfs, ignore_index=True)

        # 按时间戳排序
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

        print(f"[OrderbookLoader] 成功合并 {len(dfs)} 天数据: "
              f"总计 {len(combined_df):,}行, {len(combined_df['timestamp'].unique())}个截面")

        return combined_df

    def get_snapshot(self, df: pd.DataFrame, timestamp: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        提取指定时刻的orderbook截面

        参数：
            df: orderbook DataFrame
            timestamp: 时间戳（毫秒）

        返回：
            (bid_book, ask_book)
            - bid_book: DataFrame，按price降序排列
            - ask_book: DataFrame，按price升序排列
        """
        snapshot = df[df['timestamp'] == timestamp]

        if len(snapshot) == 0:
            raise ValueError(f"时间戳 {timestamp} 不存在于数据中")

        # 分离bid和ask
        bid_book = snapshot[snapshot['side'] == 'bid'].copy()
        ask_book = snapshot[snapshot['side'] == 'ask'].copy()

        # 排序
        bid_book = bid_book.sort_values('price', ascending=False).reset_index(drop=True)
        ask_book = ask_book.sort_values('price', ascending=True).reset_index(drop=True)

        return bid_book, ask_book

    def get_best_prices(self, df: pd.DataFrame, timestamp: int) -> Tuple[float, float]:
        """
        获取指定时刻的最优买卖价

        参数：
            df: orderbook DataFrame
            timestamp: 时间戳（毫秒）

        返回：
            (best_bid, best_ask)
        """
        bid_book, ask_book = self.get_snapshot(df, timestamp)

        if len(bid_book) == 0 or len(ask_book) == 0:
            raise ValueError(f"时间戳 {timestamp} 的orderbook不完整")

        best_bid = bid_book.iloc[0]['price']
        best_ask = ask_book.iloc[0]['price']

        return best_bid, best_ask


def load_orderbook_snapshot(symbol: str, date: str,
                            data_dir: str = 'sample_data/orderbook_sanpshots',
                            filter_invalid: bool = True,
                            convert_to_ms: bool = True) -> pd.DataFrame:
    """
    便捷函数：加载单日orderbook截面数据

    参数：
        symbol: 交易对符号，如'ASTERUSDT'
        date: 日期字符串，格式'YYYY-MM-DD'
        data_dir: 数据目录路径
        filter_invalid: 是否过滤无效数据
        convert_to_ms: 是否转换时间戳为毫秒

    返回：
        DataFrame
    """
    loader = OrderbookLoader(data_dir)
    return loader.load(symbol, date, filter_invalid, convert_to_ms)
