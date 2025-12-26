# -*- coding: utf-8 -*-
"""
技术指标计算接口
高层封装，负责数据预处理和后处理
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path
import time

from .indicators import (compute_indicators_batch, compute_indicators_fixed_volatility,
                        compute_indicators_fixed_volatility_multi,
                        get_default_parameters, parse_time_string)
from .guard import compute_guard_batch
from ..utils.ticksize_manager import get_ticksize_manager


class IndicatorCalculator:
    """
    技术指标计算器

    功能：
    1. 管理计算参数
    2. 数据预处理
    3. 批量指标计算
    4. 结果后处理
    5. 数据存储
    """

    def __init__(self,
                 tau_p='5min',  # PEMA时间常数，支持字符串格式
                 tau_o='5min',  # 波动率时间常数，支持字符串格式（仅动态波动率模式）
                 sigma_multi=1.0,  # 波动率缩放因子（仅动态波动率模式）
                 fixed_volatility=None,  # 固定波动率（如0.001表示千分之一）
                 min_volatility=0.001,  # 波动率下限（默认千分之一，仅动态波动率模式）
                 guard_k=100,  # Guard窗口大小
                 time_unit=1000.0,
                 eps=1e-12,
                 dt_min=1e-3):
        """
        初始化指标计算器

        参数：
            tau_p: PEMA时间常数，支持字符串如'5min'、'3h'、'100s'
            tau_o: 波动率时间常数，支持字符串如'5min'、'3h'、'100s'（仅动态波动率模式）
            sigma_multi: 波动率缩放因子（仅动态波动率模式）
            fixed_volatility: 固定波动率，如0.001(千分之一)、0.002(千分之二)
                            None表示使用动态波动率模式
            min_volatility: 波动率下限（默认0.001即千分之一，仅动态波动率模式有效）
                          防止动态计算的波动率过小导致价差过窄
            guard_k: Guard窗口大小
            time_unit: 时间单位转换因子（毫秒转秒用1000.0）
            eps: 数值防护参数
            dt_min: 最小时间间隔（秒）
        """
        # 解析时间常数
        self.tau_p = parse_time_string(tau_p)
        self.tau_o = parse_time_string(tau_o) if fixed_volatility is None else None
        self.sigma_multi = sigma_multi if fixed_volatility is None else None
        self.min_volatility = min_volatility if fixed_volatility is None else None
        self.fixed_volatility = fixed_volatility
        self.guard_k = guard_k
        self.time_unit = time_unit
        self.eps = eps
        self.dt_min = dt_min

        # 获取ticksize管理器
        self.ticksize_manager = get_ticksize_manager()

        print(f"📊 指标计算器初始化完成")
        print(f"   PEMA时间常数: tau_p={self.tau_p}秒 (输入: {tau_p})")

        if fixed_volatility is not None:
            print(f"   波动率模式: 固定波动率")
            print(f"   固定波动率: {fixed_volatility*100:.3f}% ({fixed_volatility*1000:.1f}‰)")
        else:
            print(f"   波动率模式: 动态波动率")
            print(f"   波动率时间常数: tau_o={self.tau_o}秒 (输入: {tau_o})")
            print(f"   波动率缩放: sigma_multi={sigma_multi}")
            print(f"   波动率下限: {min_volatility*100:.3f}% ({min_volatility*1000:.1f}‰)")

        print(f"   Guard窗口: K={guard_k}")
        print(f"   时间单位: {time_unit} (毫秒转秒)")

    @classmethod
    def from_defaults(cls):
        """使用默认参数创建计算器"""
        params = get_default_parameters()
        return cls(**params)

    def load_trades_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        加载trades数据（简化版，绕过pydantic依赖）

        参数：
            file_path: ZIP文件路径

        返回：
            处理后的DataFrame
        """
        import zipfile

        print(f"🔄 加载trades数据: {file_path}")

        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                file_list = zf.namelist()
                if not file_list:
                    print("❌ ZIP文件为空")
                    return None

                data_file = file_list[0]
                print(f"   读取文件: {data_file}")

                with zf.open(data_file, 'r') as f:
                    df = pd.read_csv(f)

                print(f"✅ 原始数据: {len(df)} 行, {len(df.columns)} 列")

                # 数据预处理
                df = self._preprocess_trades(df)

                if df is not None:
                    print(f"✅ 预处理完成: {len(df)} 行")
                    return df
                else:
                    print("❌ 预处理失败")
                    return None

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None

    def _preprocess_trades(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        预处理trades数据

        参数：
            df: 原始DataFrame

        返回：
            预处理后的DataFrame
        """
        # 检查必需字段
        required_fields = ['agg_trade_id', 'price', 'quantity', 'transact_time', 'is_buyer_maker']
        missing_fields = [field for field in required_fields if field not in df.columns]

        if missing_fields:
            print(f"❌ 缺少必需字段: {missing_fields}")
            return None

        # 重命名和标准化字段
        df_processed = df.copy()

        # 统一字段名
        df_processed['timestamp'] = df_processed['transact_time']  # 毫秒时间戳
        df_processed['trade_price'] = df_processed['price']
        df_processed['trade_qty'] = df_processed['quantity']
        df_processed['trade_id'] = df_processed['agg_trade_id']

        # 按时间戳排序
        df_processed = df_processed.sort_values('timestamp').reset_index(drop=True)

        # 检查数据质量
        if len(df_processed) == 0:
            print("❌ 处理后数据为空")
            return None

        if (df_processed['trade_price'] <= 0).any():
            print("❌ 发现非正价格")
            return None

        if (df_processed['trade_qty'] <= 0).any():
            print("❌ 发现非正数量")
            return None

        # 打印数据摘要
        print(f"   时间范围: {df_processed['timestamp'].min()} - {df_processed['timestamp'].max()}")
        print(f"   价格范围: {df_processed['trade_price'].min():.2f} - {df_processed['trade_price'].max():.2f}")
        print(f"   总成交量: {df_processed['trade_qty'].sum():.4f}")

        return df_processed

    def calculate_indicators(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        计算技术指标和Guard价位

        参数：
            df: 预处理后的trades数据
            symbol: 交易品种（用于获取ticksize）

        返回：
            包含指标和Guard的完整DataFrame
        """
        print(f"🧮 开始指标计算...")
        print(f"   数据量: {len(df)} 行")

        start_time = time.time()

        # 获取ticksize
        if symbol:
            ticksize = self.ticksize_manager.get_or_detect(symbol,
                df['trade_price'].values if 'trade_price' in df.columns else df['price'].values)
        else:
            ticksize = df.attrs.get('ticksize', 0.01)

        print(f"   使用ticksize: {ticksize}")

        # 准备numba输入数组（适配实际字段名）
        # 时间戳：优先使用ts_exch，否则使用transact_time
        if 'ts_exch' in df.columns:
            timestamps = df['ts_exch'].values.astype(np.float64)
        elif 'timestamp' in df.columns:
            timestamps = df['timestamp'].values.astype(np.float64)
        else:
            timestamps = df['transact_time'].values.astype(np.float64)

        # 价格：优先使用price
        if 'price' in df.columns:
            prices = df['price'].values.astype(np.float64)
        else:
            prices = df['trade_price'].values.astype(np.float64)

        # 数量：优先使用qty
        if 'qty' in df.columns:
            volumes = df['qty'].values.astype(np.float64)
        else:
            volumes = df['trade_qty'].values.astype(np.float64)

        # 根据波动率模式调用不同的计算函数
        if self.fixed_volatility is not None:
            # 固定波动率模式
            result = compute_indicators_fixed_volatility(
                timestamps, prices, volumes,
                self.tau_p, self.fixed_volatility, ticksize,
                self.time_unit, self.eps, self.dt_min
            )
            # 解包结果（固定波动率模式返回7个值）
            (vema, pema_prior, pema, fair, ask_permit, bid_permit, sigma) = result
            # 为兼容性创建空的动态指标数组
            vsqema = np.zeros_like(pema)
            vema_o = np.zeros_like(pema)
            msq_per_trade = np.zeros_like(pema)
        else:
            # 动态波动率模式
            result = compute_indicators_batch(
                timestamps, prices, volumes,
                self.tau_p, self.tau_o, self.sigma_multi, self.min_volatility, ticksize,
                self.time_unit, self.eps, self.dt_min
            )
            # 解包结果（动态波动率模式返回10个值）
            (vema, pema_prior, pema, vsqema, vema_o,
             msq_per_trade, sigma, fair, ask_permit, bid_permit) = result

        # 计算Guard价位
        print(f"🛡️ 计算Guard价位...")
        is_buyer_maker = df['is_buyer_maker'].values if 'is_buyer_maker' in df.columns else np.zeros(len(df), dtype=bool)

        guard_ask, guard_bid = compute_guard_batch(
            timestamps.astype(np.int64),
            prices,
            is_buyer_maker,
            self.guard_k,
            ticksize
        )

        # 创建增强的DataFrame
        df_enhanced = df.copy()

        # 添加技术指标列
        df_enhanced['vema'] = vema
        df_enhanced['pema_prior'] = pema_prior
        df_enhanced['pema'] = pema
        df_enhanced['vsqema'] = vsqema
        df_enhanced['vema_o'] = vema_o
        df_enhanced['msq_per_trade'] = msq_per_trade
        df_enhanced['sigma'] = sigma
        df_enhanced['fair'] = fair
        df_enhanced['ask_permit'] = ask_permit
        df_enhanced['bid_permit'] = bid_permit

        # 添加Guard列
        df_enhanced['guard_ask'] = guard_ask
        df_enhanced['guard_bid'] = guard_bid

        # 保留ticksize信息
        df_enhanced.attrs['ticksize'] = ticksize
        if symbol:
            df_enhanced.attrs['symbol'] = symbol

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ 指标计算完成")
        print(f"   耗时: {duration:.3f}秒")
        print(f"   处理速度: {len(df)/duration:.0f} trades/秒")

        # 打印指标摘要
        self._print_indicator_summary(df_enhanced)

        return df_enhanced

    def _print_indicator_summary(self, df: pd.DataFrame):
        """打印指标摘要统计"""
        print(f"\n📈 指标摘要:")
        print(f"   VEMA范围: {df['vema'].min():.6f} - {df['vema'].max():.6f}")
        print(f"   PEMA范围: {df['pema'].min():.2f} - {df['pema'].max():.2f}")
        print(f"   波动率σ范围: {df['sigma'].min():.4f} - {df['sigma'].max():.4f}")
        print(f"   挂单价差范围: {(df['ask_permit'] - df['bid_permit']).min():.2f} - {(df['ask_permit'] - df['bid_permit']).max():.2f}")
        print(f"   Guard_ask范围: {df['guard_ask'].min():.2f} - {df['guard_ask'].max():.2f}")
        print(f"   Guard_bid范围: {df['guard_bid'].min():.2f} - {df['guard_bid'].max():.2f}")
        print(f"   Guard价差范围: {(df['guard_ask'] - df['guard_bid']).min():.2f} - {(df['guard_ask'] - df['guard_bid']).max():.2f}")

    def split_by_hours(self, df: pd.DataFrame, hours: int = 4) -> Dict[str, pd.DataFrame]:
        """
        按小时切分数据

        参数：
            df: 完整数据
            hours: 每个文件包含的小时数

        返回：
            时间段标签 -> DataFrame的字典
        """
        print(f"⏰ 按{hours}小时切分数据...")

        # 转换时间戳为datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 计算时间段
        start_time = df['datetime'].min()
        end_time = df['datetime'].max()

        print(f"   时间范围: {start_time} - {end_time}")

        # 生成时间段
        time_segments = pd.date_range(
            start=start_time.floor(f'{hours}H'),
            end=end_time.ceil(f'{hours}H'),
            freq=f'{hours}H'
        )

        segments = {}

        for i in range(len(time_segments) - 1):
            segment_start = time_segments[i]
            segment_end = time_segments[i + 1]

            # 筛选数据
            mask = (df['datetime'] >= segment_start) & (df['datetime'] < segment_end)
            segment_data = df[mask].copy()

            if len(segment_data) > 0:
                # 生成标签
                label = segment_start.strftime('%Y%m%d_%H%M')
                segments[label] = segment_data

                print(f"   时间段 {label}: {len(segment_data)} 行")

        print(f"✅ 数据切分完成，共{len(segments)}个时间段")
        return segments

    def save_segments(self, segments: Dict[str, pd.DataFrame], output_dir: str) -> bool:
        """
        保存时间段数据

        参数：
            segments: 时间段数据字典
            output_dir: 输出目录

        返回：
            是否成功
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"💾 保存数据到: {output_path}")

        try:
            saved_count = 0

            for label, segment_df in segments.items():
                filename = f"indicators_{label}.csv"
                file_path = output_path / filename

                # 选择要保存的列
                columns_to_save = [
                    'timestamp', 'trade_price', 'trade_qty', 'trade_id',
                    'vema', 'pema_prior', 'pema', 'vsqema', 'vema_o',
                    'msq_per_trade', 'sigma', 'fair', 'ask_permit', 'bid_permit'
                ]

                # 确保列存在
                available_columns = [col for col in columns_to_save if col in segment_df.columns]

                # 保存CSV
                segment_df[available_columns].to_csv(file_path, index=False, float_format='%.8f')

                file_size_mb = file_path.stat().st_size / 1024 / 1024
                print(f"   ✅ {filename}: {len(segment_df)} 行, {file_size_mb:.2f} MB")

                saved_count += 1

            print(f"✅ 保存完成，共{saved_count}个文件")
            return True

        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False

    def process_file(self, input_file: str, output_dir: str, hours: int = 4) -> bool:
        """
        完整处理流程

        参数：
            input_file: 输入ZIP文件路径
            output_dir: 输出目录
            hours: 每个文件的小时数

        返回：
            是否成功
        """
        print(f"🚀 开始完整处理流程")
        print(f"   输入: {input_file}")
        print(f"   输出: {output_dir}")
        print(f"   时间切分: {hours}小时/文件")
        print()

        # 1. 加载数据
        df = self.load_trades_data(input_file)
        if df is None:
            return False

        print()

        # 2. 计算指标
        df_enhanced = self.calculate_indicators(df)

        print()

        # 3. 切分数据
        segments = self.split_by_hours(df_enhanced, hours)

        print()

        # 4. 保存数据
        success = self.save_segments(segments, output_dir)

        if success:
            print(f"\n🎉 处理完成！")
            print(f"📁 输出目录: {output_dir}")
            print(f"📊 生成文件: {len(segments)} 个")

        return success