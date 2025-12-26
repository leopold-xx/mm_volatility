# -*- coding: utf-8 -*-
"""
Trades数据加载器
"""

import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from ..utils.time_utils import to_ms_timestamp
from ..utils.ticksize_manager import get_ticksize_manager
from .schema import validate_trades_dataframe
from pathlib import Path
from typing import Optional, Tuple, Sequence
import pandas as pd
import polars as pl

class TradesLoader:
    """
    Binance aggTrades数据加载器
    处理ZIP压缩的CSV格式交易数据
    集成Ticksize管理
    """

    def __init__(self, auto_detect_ticksize: bool = True):
        """
        初始化数据加载器

        参数：
            auto_detect_ticksize: 是否自动检测ticksize
        """
        self.auto_detect_ticksize = auto_detect_ticksize
        self.ticksize_manager = get_ticksize_manager()
    
    def load_from_zip(self, zip_path: str, symbol: str = None) -> Tuple[Optional[pd.DataFrame], float]:
        """
        从ZIP文件加载trades数据，应用ticksize处理

        Args:
            zip_path: ZIP文件路径
            symbol: 交易品种（用于ticksize管理）

        Returns:
            (处理后的DataFrame, ticksize)，失败返回(None, 0.01)
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"文件不存在: {zip_path}")

        # 如果没有提供symbol，尝试从文件名提取
        if symbol is None:
            # 例如：BTCUSDT-aggTrades-2024-07-08.zip
            filename = zip_path.stem  # 去掉.zip
            parts = filename.split('-')
            if len(parts) >= 1:
                symbol = parts[0].lower()
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                file_list = zf.namelist()
                if not file_list:
                    raise ValueError("ZIP文件为空")
                
                # 通常只有一个CSV文件
                data_file = file_list[0]
                
                with zf.open(data_file, 'r') as f:
                    df = pd.read_csv(f)
                
                # 数据校验
                validate_trades_dataframe(df)

                # 获取或检测ticksize
                if symbol and self.auto_detect_ticksize:
                    ticksize = self.ticksize_manager.get_or_detect(symbol, df['price'].values)
                else:
                    ticksize = self.ticksize_manager.get_ticksize(symbol) or 0.01

                print(f"📊 应用ticksize: {symbol} = {ticksize}")

                # 应用ticksize到价格（向上进位，避免乐观）
                df['price'] = self.ticksize_manager.round_prices(
                    df['price'].values, ticksize, direction='up'
                )

                # 时间戳统一转换为毫秒
                df['ts_exch'] = df['transact_time'].apply(to_ms_timestamp)
                df['ts_local'] = df['ts_exch']  # 默认与交易所时间相同

                # 统一字段命名：quantity -> qty
                if 'quantity' in df.columns:
                    df = df.rename(columns={'quantity': 'qty'})

                # 统一side字段：is_buyer_maker -> side
                df['side'] = df['is_buyer_maker'].map({
                    True: 'sell',   # 买方是maker，说明是sell taker成交
                    False: 'buy'    # 买方是taker，说明是buy taker成交
                })

                # 添加ticksize信息到DataFrame属性
                df.attrs['ticksize'] = ticksize
                df.attrs['symbol'] = symbol

                return df, ticksize
                
        except Exception as e:
            print(f"加载trades数据失败: {e}")
            return None, 0.01
    
    def load_date_range(self, symbol: str, start_date: str, end_date: str,
                       base_path: str = "/mnt/nas/cta_public/MarketData/BinanceSource") -> Tuple[pd.DataFrame, float]:
        """
        加载指定日期范围的trades数据

        Args:
            symbol: 交易对，如'btcusdt'
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            base_path: 数据根路径

        Returns:
            (合并后的DataFrame, ticksize)
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        all_trades = []
        ticksize = None

        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            symbol_upper = symbol.upper()

            zip_path = Path(base_path) / f"binance_usd_{symbol}" / "aggtrades" / f"{symbol_upper}-aggTrades-{date_str}.zip"

            print(f"正在加载: {zip_path}")
            df_day, day_ticksize = self.load_from_zip(str(zip_path), symbol)

            if df_day is not None:
                all_trades.append(df_day)
                if ticksize is None:
                    ticksize = day_ticksize
            else:
                print(f"跳过日期: {date_str}")

        if not all_trades:
            raise ValueError("没有加载到任何trades数据")

        # 合并所有日期数据
        df_combined = pd.concat(all_trades, ignore_index=True)

        # 按时间戳排序
        df_combined = df_combined.sort_values('ts_exch').reset_index(drop=True)

        # 保留ticksize和symbol信息
        df_combined.attrs['ticksize'] = ticksize
        df_combined.attrs['symbol'] = symbol

        return df_combined, ticksize

    

    def load_from_parquet(
        self,
        parquet_path: str,
        symbol: str = None
    ) -> Tuple[Optional[pd.DataFrame], float]:
        """
        从Parquet加载trades数据（polars加速），应用ticksize处理
        - 在 validate_trades_dataframe(df) 前转 pandas
        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"文件不存在: {parquet_path}")

        # 如果没有提供symbol，尝试从文件名提取
        if symbol is None:
            # e.g. BTCUSDT-aggTrades-2024-07-08.parquet
            parts = parquet_path.stem.split("-")
            if len(parts) >= 1:
                symbol = parts[0].lower()

        try:
            # ---- 1) scan parquet（懒加载）----
            lf = pl.read_parquet(str(parquet_path))
            df = lf.to_pandas()

            # 数据校验
            validate_trades_dataframe(df)

            # 获取或检测ticksize
            if symbol and self.auto_detect_ticksize:
                ticksize = self.ticksize_manager.get_or_detect(symbol, df['price'].values)
            else:
                ticksize = self.ticksize_manager.get_ticksize(symbol) or 0.01

            print(f"📊 应用ticksize: {symbol} = {ticksize}")

            # 应用ticksize到价格（向上进位，避免乐观）
            df['price'] = self.ticksize_manager.round_prices(
                df['price'].values, ticksize, direction='up'
            )

            # 时间戳统一转换为毫秒
            df['ts_exch'] = df['transact_time'].apply(to_ms_timestamp)
            df['ts_local'] = df['ts_exch']  # 默认与交易所时间相同

            # 统一字段命名：quantity -> qty
            if 'quantity' in df.columns:
                df = df.rename(columns={'quantity': 'qty'})

            # 统一side字段：is_buyer_maker -> side
            df['side'] = df['is_buyer_maker'].map({
                True: 'sell',   # 买方是maker，说明是sell taker成交
                False: 'buy'    # 买方是taker，说明是buy taker成交
            })

            # 添加ticksize信息到DataFrame属性
            df.attrs['ticksize'] = ticksize
            df.attrs['symbol'] = symbol

            return df, ticksize
            
        except Exception as e:
            print(f"加载trades数据失败: {e}")
            return None, 0.01