# -*- coding: utf-8 -*-
"""
动态订单大小计算工具
根据品种价格和初始资金动态计算合适的订单大小
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path

from mmcore.io.trades_loader import TradesLoader


class OrderSizeCalculator:
    """动态订单大小计算器"""

    def __init__(self, initial_cash: float = 100000.0):
        """
        初始化订单大小计算器

        参数:
            initial_cash: 初始资金
        """
        self.initial_cash = initial_cash
        self.size_multipliers = [0.01, 0.1, 1, 10, 100, 1000, 10000,
                                 100000, 1000000, 1000000]
        self.price_cache = {}  # 缓存品种价格

    def get_symbol_price(self, symbol: str, data_file_path: str) -> float:
        """
        获取品种的代表性价格（第一个日期的成交均价）

        参数:
            symbol: 交易品种
            data_file_path: 数据文件路径

        返回:
            代表性价格
        """
        # 检查缓存
        if symbol in self.price_cache:
            return self.price_cache[symbol]

        try:
            # 加载数据
            loader = TradesLoader()
            df, _ = loader.load_from_zip(data_file_path)

            if df is None or len(df) == 0:
                raise ValueError(f"无法加载数据或数据为空: {data_file_path}")

            # 计算成交量加权平均价格(VWAP)
            if 'price' in df.columns and 'qty' in df.columns:
                price_col = 'price'
                qty_col = 'qty'
            else:
                price_col = 'trade_price'
                qty_col = 'trade_qty'

            total_value = (df[price_col] * df[qty_col]).sum()
            total_volume = df[qty_col].sum()

            if total_volume == 0:
                # 如果没有成交量，使用简单平均价
                avg_price = df[price_col].mean()
            else:
                # 使用成交量加权平均价
                avg_price = total_value / total_volume

            # 缓存结果
            self.price_cache[symbol] = avg_price

            return avg_price

        except Exception as e:
            print(f"⚠️ 获取 {symbol} 价格失败: {e}")
            # 返回默认价格（基于常见价格范围估算）
            default_prices = {
                'BTCUSDT': 60000.0,
                'ETHUSDT': 3000.0,
                'XRPUSDT': 0.5,
                'ADAUSDT': 0.4,
                'SOLUSDT': 150.0,
                'DOGEUSDT': 0.07,
                'AVAXUSDT': 25.0,
                'LINKUSDT': 10.0,
                'DOTUSDT': 5.0,
                'MATICUSDT': 0.8,
                'LTCUSDT': 70.0,
                'BCHUSDT': 300.0,
            }

            default_price = default_prices.get(symbol, 1.0)
            self.price_cache[symbol] = default_price
            return default_price

    def calculate_optimal_order_size(self, symbol: str, price: float) -> float:
        """
        计算最优订单大小

        选择使得 price × order_size 最接近 initial_cash 的 order_size

        参数:
            symbol: 交易品种
            price: 品种价格

        返回:
            最优订单大小
        """
        target_value = self.initial_cash

        # 计算每个倍数对应的订单价值
        best_size = self.size_multipliers[0]
        best_diff = float('inf')

        for multiplier in self.size_multipliers:
            order_value = price * multiplier
            diff = abs(order_value - target_value)

            if diff < best_diff:
                best_diff = diff
                best_size = multiplier

        return best_size

    def calculate_symbol_order_sizes(self, symbols: List[str],
                                   data_scanner,
                                   start_date: str) -> Dict[str, float]:
        """
        批量计算多个品种的订单大小

        参数:
            symbols: 品种列表
            data_scanner: 数据扫描器
            start_date: 开始日期（用于获取第一个可用日期的数据）

        返回:
            品种订单大小映射字典
        """
        symbol_order_sizes = {}

        print(f"📊 计算动态订单大小 (初始资金: ${self.initial_cash:,.0f})")
        print("=" * 60)

        for symbol in symbols:
            try:
                # 获取该品种的可用日期
                available_dates = data_scanner.scan_symbol_dates(symbol)

                if not available_dates:
                    print(f"⚠️ {symbol}: 无可用数据")
                    # 使用默认价格计算
                    default_price = self.get_symbol_price(symbol, "")
                    order_size = self.calculate_optimal_order_size(symbol, default_price)
                    symbol_order_sizes[symbol] = order_size
                    continue

                # 使用最早的可用日期
                first_date = min(available_dates)

                # 获取数据文件路径
                data_file_path = data_scanner.get_file_path(symbol, first_date)

                # 获取价格
                price = self.get_symbol_price(symbol, str(data_file_path))

                # 计算订单大小
                order_size = self.calculate_optimal_order_size(symbol, price)

                symbol_order_sizes[symbol] = order_size

                # 计算订单价值
                order_value = price * order_size
                value_ratio = order_value / self.initial_cash

                print(f"{symbol:12}: "
                      f"价格 ${price:8.4f}, "
                      f"订单 {order_size:8.0f}, "
                      f"价值 ${order_value:8.0f} "
                      f"({value_ratio:.1%} 资金)")

            except Exception as e:
                print(f"❌ {symbol}: 计算失败 - {e}")
                # 使用最小的默认订单大小
                symbol_order_sizes[symbol] = self.size_multipliers[0]

        return symbol_order_sizes

    def update_param_configs_with_order_sizes(self,
                                            param_configs: List[Dict],
                                            symbol_order_sizes: Dict[str, float]) -> List[Dict]:
        """
        用动态订单大小更新参数配置

        参数:
            param_configs: 参数配置列表
            symbol_order_sizes: 品种订单大小映射

        返回:
            更新后的参数配置列表
        """
        updated_configs = []

        for config in param_configs:
            updated_config = config.copy()

            # 如果是动态订单大小类型，则添加订单大小映射
            if config.get('order_size_type') == 'dynamic':
                updated_config['symbol_order_sizes'] = symbol_order_sizes
                # 移除类型标记，避免序列化问题
                updated_config.pop('order_size_type', None)
            else:
                # 如果没有指定动态类型，使用固定订单大小（向后兼容）
                if 'order_size' not in updated_config:
                    updated_config['order_size'] = 0.01

            updated_configs.append(updated_config)

        return updated_configs

    def get_order_size_for_symbol(self, symbol: str,
                                symbol_order_sizes: Dict[str, float]) -> float:
        """
        获取指定品种的订单大小

        参数:
            symbol: 交易品种
            symbol_order_sizes: 品种订单大小映射

        返回:
            订单大小
        """
        return symbol_order_sizes.get(symbol, 0.01)  # 默认0.01

    def print_order_size_summary(self, symbol_order_sizes: Dict[str, float]):
        """打印订单大小汇总"""
        print(f"\n📋 订单大小汇总:")
        print("-" * 60)

        # 按订单大小排序
        sorted_symbols = sorted(
            symbol_order_sizes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for symbol, order_size in sorted_symbols:
            # 估算价格（从缓存获取）
            price = self.price_cache.get(symbol, 1.0)
            order_value = price * order_size
            value_ratio = order_value / self.initial_cash

            print(f"{symbol:12}: {order_size:8.0f} "
                  f"(价值 ${order_value:8.0f}, {value_ratio:.1%} 资金)")

        # 统计信息
        sizes = list(symbol_order_sizes.values())
        print(f"\n📊 统计信息:")
        print(f"   品种数量: {len(sizes)}")
        print(f"   订单大小范围: {min(sizes)} ~ {max(sizes)}")
        print(f"   平均订单大小: {np.mean(sizes):.2f}")

        # 按订单大小分组
        size_groups = {}
        for symbol, size in symbol_order_sizes.items():
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(symbol)

        print(f"\n📦 分组统计:")
        for size in sorted(size_groups.keys(), reverse=True):
            symbols = size_groups[size]
            print(f"   订单大小 {size:6.0f}: {len(symbols):2}个品种 {symbols}")


def calculate_and_update_order_sizes(config_module, data_scanner) -> Dict[str, float]:
    """
    计算并更新所有参数配置的订单大小

    参数:
        config_module: 配置模块
        data_scanner: 数据扫描器

    返回:
        品种订单大小映射
    """
    print("🔧 开始动态订单大小计算...")

    # 创建计算器
    calculator = OrderSizeCalculator(
        initial_cash=config_module.BACKTEST_CONFIG['initial_cash']
    )

    # 计算各品种订单大小
    symbol_order_sizes = calculator.calculate_symbol_order_sizes(
        symbols=config_module.ACTIVE_SYMBOLS,
        data_scanner=data_scanner,
        start_date=config_module.DATE_RANGE_CONFIG['start_date']
    )

    # 打印汇总
    calculator.print_order_size_summary(symbol_order_sizes)

    # 更新参数配置
    print(f"\n🔄 更新参数配置...")
    config_module.ACTIVE_PARAM_CONFIGS = calculator.update_param_configs_with_order_sizes(
        param_configs=config_module.ACTIVE_PARAM_CONFIGS,
        symbol_order_sizes=symbol_order_sizes
    )

    print(f"✅ 动态订单大小计算完成")

    return symbol_order_sizes


if __name__ == "__main__":
    # 测试订单大小计算
    calculator = OrderSizeCalculator(initial_cash=100000.0)

    # 测试一些典型价格
    test_cases = [
        ('BTCUSDT', 60000.0),
        ('ETHUSDT', 3000.0),
        ('XRPUSDT', 0.5),
        ('DOGEUSDT', 0.07),
    ]

    print("🧪 订单大小计算测试:")
    print("=" * 50)

    for symbol, price in test_cases:
        order_size = calculator.calculate_optimal_order_size(symbol, price)
        order_value = price * order_size
        value_ratio = order_value / calculator.initial_cash

        print(f"{symbol:10}: 价格 ${price:8.2f}, "
              f"订单 {order_size:8.0f}, "
              f"价值 ${order_value:8.0f} "
              f"({value_ratio:.1%})")