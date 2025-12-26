# -*- coding: utf-8 -*-
"""
统一数据加载器
加载trades数据，生成回测就绪的数据集
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from .trades_loader import TradesLoader


class UnifiedDataLoader:
    """
    统一数据加载器
    加载和处理trades数据
    """

    def __init__(self):
        self.trades_loader = TradesLoader()

    def load_single_day(self, symbol: str, date: str,
                       base_path: str = "/mnt/nas/cta_public/MarketData") -> Optional[pd.DataFrame]:
        """
        加载单日trades数据

        Args:
            symbol: 交易对，如'btcusdt'
            date: 日期 'YYYY-MM-DD'
            base_path: 数据根路径

        Returns:
            处理后的trades DataFrame
        """
        print(f"🔄 加载 {symbol.upper()} {date} trades数据")
        print("=" * 50)

        # 加载trades数据
        print("📊 加载trades数据...")
        trades_path = self._get_trades_path(symbol, date, base_path)
        df_trades = self.trades_loader.load_from_zip(trades_path)

        if df_trades is None:
            print(f"❌ trades数据加载失败: {trades_path}")
            return None

        print(f"✅ trades数据: {len(df_trades)} 行")

        # 创建统一的数据结构
        df_unified = self._create_unified_structure(df_trades)

        print(f"✅ 数据处理完成: {len(df_unified)} 行")
        print()

        return df_unified

    def _get_trades_path(self, symbol: str, date: str, base_path: str) -> str:
        """构建trades数据路径"""
        symbol_upper = symbol.upper()
        return str(Path(base_path) / "BinanceSource" / f"binance_usd_{symbol}" /
                  "aggtrades" / f"{symbol_upper}-aggTrades-{date}.zip")

    def _create_unified_structure(self, df_trades: pd.DataFrame) -> pd.DataFrame:
        """
        创建统一的数据结构
        """
        # 确保时间戳列存在
        if 'ts_exch' not in df_trades.columns:
            raise ValueError("缺少时间戳字段 ts_exch")

        # 按时间戳排序
        df_trades = df_trades.sort_values('ts_exch').reset_index(drop=True)

        print(f"  trades时间范围: {df_trades['ts_exch'].min()} - {df_trades['ts_exch'].max()}")

        # 创建精简的统一结构
        df_unified = pd.DataFrame({
            # 时间信息（只保留服务器时间戳）
            'timestamp': df_trades['ts_exch'],

            # 交易信息
            'trade_price': df_trades['price'],
            'trade_qty': df_trades['qty'],
            'trade_side': df_trades['side'],
            'trade_id': df_trades.get('agg_trade_id', 0),
            'is_buyer_maker': df_trades['is_buyer_maker'],
        })

        # 按时间戳和trade_id排序（相同时间戳按trade_id排序）
        df_unified = df_unified.sort_values(['timestamp', 'trade_id']).reset_index(drop=True)

        return df_unified

    def save_unified_data(self, df: pd.DataFrame, output_path: str) -> bool:
        """
        保存统一格式的数据

        Args:
            df: 统一格式的DataFrame
            output_path: 输出文件路径

        Returns:
            保存是否成功
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存为CSV格式
            df.to_csv(output_path, index=False, float_format='%.8f')

            print(f"✅ 数据已保存至: {output_path}")
            print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

            return True

        except Exception as e:
            print(f"❌ 数据保存失败: {e}")
            return False

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        生成数据摘要
        """
        summary = {
            'total_trades': len(df),
            'time_range_ms': (df['timestamp'].min(), df['timestamp'].max()),
            'duration_hours': (df['timestamp'].max() - df['timestamp'].min()) / (1000 * 3600),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'trade_price_range': (df['trade_price'].min(), df['trade_price'].max()),
            'trade_volume': df['trade_qty'].sum(),
            'unique_trade_prices': df['trade_price'].nunique(),
            'buy_trades': (df['trade_side'] == 'buy').sum(),
            'sell_trades': (df['trade_side'] == 'sell').sum(),
        }

        return summary