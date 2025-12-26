# -*- coding: utf-8 -*-
"""
数据格式定义和校验
"""

from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np


class TradeRecord(BaseModel):
    """单笔交易记录"""
    agg_trade_id: int = Field(description="聚合交易ID")
    price: float = Field(gt=0, description="成交价格")
    quantity: float = Field(gt=0, description="成交数量")
    first_trade_id: int = Field(description="聚合的第一笔交易ID")
    last_trade_id: int = Field(description="聚合的最后一笔交易ID")
    transact_time: int = Field(description="交易时间戳（毫秒）")
    is_buyer_maker: bool = Field(description="买方是否为maker")


def validate_trades_dataframe(df: pd.DataFrame) -> bool:
    """
    校验trades DataFrame格式
    """
    required_cols = ['agg_trade_id', 'price', 'quantity', 'first_trade_id',
                     'last_trade_id', 'transact_time', 'is_buyer_maker']

    if not all(col in df.columns for col in required_cols):
        missing = set(required_cols) - set(df.columns)
        raise ValueError(f"缺失必需字段: {missing}")

    # 检查数据类型和范围
    if (df['price'] <= 0).any():
        raise ValueError("价格必须为正数")

    if (df['quantity'] <= 0).any():
        raise ValueError("数量必须为正数")

    return True