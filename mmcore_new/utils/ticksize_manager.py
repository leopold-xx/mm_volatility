# -*- coding: utf-8 -*-
"""
Ticksize管理器
负责品种ticksize的获取、检测、缓存和应用
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional
from numba import jit


class TicksizeManager:
    """
    Ticksize管理器

    功能：
    1. 维护品种->ticksize映射字典
    2. 自动检测未知品种的ticksize
    3. 缓存ticksize避免重复计算
    4. 提供价格进位函数
    """

    def __init__(self, config_file: str = None):
        """
        初始化Ticksize管理器

        参数：
            config_file: 配置文件路径，默认为项目根目录下的ticksize_config.json
        """
        # 初始化为空字典，所有品种信息从配置文件加载
        self.ticksize_dict = {}

        # 配置文件路径
        if config_file is None:
            # 查找项目根目录（包含CLAUDE.md的目录）
            project_root = self._find_project_root()
            config_file = project_root / 'ticksize_config.json'

        self.config_file = Path(config_file)

        # 确保配置文件存在
        self._ensure_config_file()

        # 加载配置
        self.load_config()

    def _find_project_root(self) -> Path:
        """查找项目根目录（包含CLAUDE.md的目录）"""
        current_path = Path(__file__).parent
        while current_path != current_path.parent:
            if (current_path / 'CLAUDE.md').exists():
                return current_path
            current_path = current_path.parent
        # 如果找不到，返回当前工作目录
        return Path.cwd()

    def _ensure_config_file(self):
        """确保配置文件存在，如果不存在则创建空文件"""
        if not self.config_file.exists():
            # 创建父目录
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            # 创建空的JSON文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2, ensure_ascii=False)
            print(f"📝 创建新的ticksize配置文件: {self.config_file}")

    def load_config(self):
        """从配置文件加载ticksize字典"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.ticksize_dict = json.load(f)
                if self.ticksize_dict:
                    print(f"✅ 加载ticksize配置: {len(self.ticksize_dict)}个品种")
                else:
                    print(f"📋 ticksize配置文件为空，将随测试进程自动添加品种信息")
        except Exception as e:
            print(f"⚠️ 加载ticksize配置失败: {e}")
            self.ticksize_dict = {}

    def save_config(self):
        """保存ticksize字典到配置文件"""
        try:
            # 按品种名称排序以保持文件整洁
            sorted_dict = dict(sorted(self.ticksize_dict.items()))
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(sorted_dict, f, indent=2, ensure_ascii=False)
                print(f"💾 保存ticksize配置: {len(sorted_dict)}个品种到 {self.config_file}")
        except Exception as e:
            print(f"❌ 保存ticksize配置失败: {e}")

    def get_ticksize(self, symbol: str) -> Optional[float]:
        """
        获取品种的ticksize

        参数：
            symbol: 品种代码（小写）

        返回：
            ticksize值，如果未找到返回None
        """
        symbol = symbol.lower()
        return self.ticksize_dict.get(symbol)

    def set_ticksize(self, symbol: str, ticksize: float, auto_save: bool = True):
        """
        设置品种的ticksize

        参数：
            symbol: 品种代码
            ticksize: ticksize值
            auto_save: 是否自动保存到配置文件
        """
        symbol = symbol.lower()
        old_ticksize = self.ticksize_dict.get(symbol)

        if old_ticksize != ticksize:
            self.ticksize_dict[symbol] = ticksize
            if auto_save:
                self.save_config()

            if old_ticksize is None:
                print(f"➕ 新增{symbol} ticksize: {ticksize}")
            else:
                print(f"🔄 更新{symbol} ticksize: {old_ticksize} -> {ticksize}")
        else:
            print(f"ℹ️ {symbol} ticksize已是: {ticksize}")

    def detect_ticksize(self, prices: np.ndarray, min_samples: int = 100) -> float:
        """
        从价格数组中自动检测ticksize

        算法：
        1. 计算所有非零价格差
        2. 找出最小的正价格差
        3. 使用最大公约数方法确定ticksize

        参数：
            prices: 价格数组
            min_samples: 最少需要的样本数

        返回：
            检测到的ticksize
        """
        if len(prices) < min_samples:
            print(f"⚠️ 样本数不足({len(prices)}<{min_samples})，使用默认ticksize")
            return 0.01

        # 使用numba加速的检测函数
        ticksize = _detect_ticksize_numba(prices)

        # 合理性检查
        if ticksize < 1e-8 or ticksize > 1000:
            print(f"⚠️ 检测到异常ticksize: {ticksize}，使用默认值")
            return 0.01

        return ticksize

    @staticmethod
    def round_to_ticksize(price: float, ticksize: float, direction: str = 'up') -> float:
        """
        将价格进位到ticksize的整数倍

        参数：
            price: 原始价格
            ticksize: tick大小
            direction: 进位方向 'up'(向上), 'down'(向下), 'nearest'(最近)

        返回：
            进位后的价格
        """
        if direction == 'up':
            return np.ceil(price / ticksize) * ticksize
        elif direction == 'down':
            return np.floor(price / ticksize) * ticksize
        else:  # nearest
            return np.round(price / ticksize) * ticksize

    def round_prices(self, prices: np.ndarray, ticksize: float,
                     direction: str = 'up') -> np.ndarray:
        """
        批量进位价格数组

        参数：
            prices: 价格数组
            ticksize: tick大小
            direction: 进位方向

        返回：
            进位后的价格数组
        """
        return _round_prices_numba(prices, ticksize, direction)

    def detect_and_save(self, symbol: str, prices: np.ndarray, auto_save: bool = True) -> float:
        """
        检测并保存品种的ticksize

        参数：
            symbol: 品种代码
            prices: 价格数组
            auto_save: 是否自动保存到配置文件

        返回：
            检测到的ticksize
        """
        print(f"🔍 正在检测{symbol.upper()}的ticksize...")
        ticksize = self.detect_ticksize(prices)
        self.set_ticksize(symbol, ticksize, auto_save=auto_save)
        return ticksize

    def get_or_detect(self, symbol: str, prices: np.ndarray = None) -> float:
        """
        获取ticksize，如果不存在则检测

        参数：
            symbol: 品种代码
            prices: 价格数组（用于检测）

        返回：
            ticksize值
        """
        symbol = symbol.lower()

        # 先尝试从字典获取
        ticksize = self.get_ticksize(symbol)
        if ticksize is not None:
            return ticksize

        # 如果没有且提供了价格数据，则检测
        if prices is not None and len(prices) > 0:
            return self.detect_and_save(symbol, prices)

        # 否则返回默认值并提示
        print(f"⚠️ 无法确定{symbol.upper()}的ticksize，使用默认值0.01")
        print(f"💡 建议提供价格数据以自动检测，或手动设置: manager.set_ticksize('{symbol}', ticksize_value)")
        return 0.01


@jit(nopython=True)
def _detect_ticksize_numba(prices: np.ndarray) -> float:
    """
    Numba加速的ticksize检测函数

    改进算法：
    1. 去重并排序价格
    2. 计算相邻价格差
    3. 统计差值频次，找出最主要的差值模式
    4. 选择最频繁出现的合理ticksize
    """
    # 去重并排序
    unique_prices = np.unique(prices)
    if len(unique_prices) < 2:
        return 0.01

    # 计算所有相邻价格差
    price_diffs = np.diff(unique_prices)

    # 找正差值
    positive_diffs = price_diffs[price_diffs > 1e-10]  # 排除浮点误差
    if len(positive_diffs) == 0:
        return 0.01

    # 将差值四舍五入到合理精度，避免浮点误差
    # 根据价格范围决定精度
    avg_price = np.mean(unique_prices)
    if avg_price > 1000:
        precision = 6  # 高价格用6位小数
    else:
        precision = 8  # 低价格用8位小数

    # 四舍五入差值
    rounded_diffs = np.round(positive_diffs, precision)

    # 统计差值频次（简化版，因为numba不支持Counter）
    # 我们寻找最常见的几个差值
    unique_diffs = np.unique(rounded_diffs)
    diff_counts = np.zeros(len(unique_diffs), dtype=np.int64)

    for i in range(len(unique_diffs)):
        diff = unique_diffs[i]
        count = 0
        for j in range(len(rounded_diffs)):
            if abs(rounded_diffs[j] - diff) < 1e-10:
                count += 1
        diff_counts[i] = count

    # 找到出现次数最多的差值
    max_count_idx = np.argmax(diff_counts)
    most_common_diff = unique_diffs[max_count_idx]
    max_count = diff_counts[max_count_idx]

    # 如果最常见差值的占比很高（>80%），就选择它
    if max_count / len(rounded_diffs) > 0.8:
        ticksize = most_common_diff
    else:
        # 否则选择最小正差值（传统方法）
        ticksize = np.min(unique_diffs)

    # 标准化到常见的ticksize值
    # 检查是否接近标准ticksize值
    standard_ticks = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])

    best_tick = ticksize
    min_error = 1.0

    for std_tick in standard_ticks:
        error = abs(ticksize - std_tick) / std_tick
        if error < min_error and error < 0.1:  # 10%误差内
            min_error = error
            best_tick = std_tick

    return best_tick


@jit(nopython=True)
def _round_prices_numba(prices: np.ndarray, ticksize: float, direction: str) -> np.ndarray:
    """
    Numba加速的批量价格进位函数
    """
    n = len(prices)
    rounded_prices = np.zeros(n, dtype=np.float64)

    if direction == 'up':
        for i in range(n):
            rounded_prices[i] = np.ceil(prices[i] / ticksize) * ticksize
    elif direction == 'down':
        for i in range(n):
            rounded_prices[i] = np.floor(prices[i] / ticksize) * ticksize
    else:  # nearest
        for i in range(n):
            rounded_prices[i] = np.round(prices[i] / ticksize) * ticksize

    return rounded_prices


# 全局单例
_global_ticksize_manager = None

def get_ticksize_manager(config_file: str = None) -> TicksizeManager:
    """获取全局Ticksize管理器单例"""
    global _global_ticksize_manager
    if _global_ticksize_manager is None:
        _global_ticksize_manager = TicksizeManager(config_file=config_file)
    return _global_ticksize_manager