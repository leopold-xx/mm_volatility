# -*- coding: utf-8 -*-
"""
时间处理工具函数
"""

import pandas as pd
from typing import Union


def to_ms_timestamp(ts: Union[int, float, str, pd.Timestamp]) -> int:
    """
    将各种时间戳格式统一转换为毫秒级整数时间戳
    
    Args:
        ts: 输入时间戳，支持秒、毫秒、微秒、纳秒整数或pandas Timestamp
    
    Returns:
        毫秒级整数时间戳
    """
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    
    if isinstance(ts, pd.Timestamp):
        return int(ts.value // 1_000_000)  # 纳秒转毫秒
    
    if isinstance(ts, (int, float)):
        # 根据数值大小判断时间戳精度
        if ts < 1e12:  # 秒级
            return int(ts * 1000)
        elif ts < 1e15:  # 毫秒级
            return int(ts)
        elif ts < 1e18:  # 微秒级
            return int(ts // 1000)
        else:  # 纳秒级
            return int(ts // 1_000_000)
    
    raise ValueError(f"不支持的时间戳格式: {type(ts)}")


def round_to_interval_ms(ts_ms: int, interval_s: int) -> int:
    """
    将毫秒时间戳向下取整到指定秒级间隔
    
    Args:
        ts_ms: 毫秒时间戳
        interval_s: 间隔秒数
    
    Returns:
        对齐到间隔的毫秒时间戳
    """
    interval_ms = interval_s * 1000
    return (ts_ms // interval_ms) * interval_ms