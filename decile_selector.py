# decile_selector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from backtest_vol_factor import build_trades_from_signal_2
from scipy.special import expit
import numpy as np

def sigmoid_scale(arr):
    arr = np.array(arr, dtype=float)
    z = (arr - arr.mean()) / (arr.std() + 1e-12)
    return expit(z)

def compute_rolling_decile(
    df: pd.DataFrame,
    time_col: str,
    factor_col: str,
    roll_decile_window: int = 90,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    只做一次 rolling decile 计算，返回带 decile 列的 df（已按时间排序）。
    decile ∈ {0..n_deciles-1}，前 window-1 行为 NaN。
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    q_edges = np.linspace(0.0, 1.0, n_deciles + 1)

    def rolling_decile_func(window_vals: np.ndarray) -> float:
        cur = window_vals[-1]
        qs = np.quantile(window_vals, q_edges)
        d = np.searchsorted(qs, cur, side="right") - 1
        d = max(0, min(n_deciles - 1, d))
        return float(d)

    df["decile"] = (
        df[factor_col]
        .rolling(window=roll_decile_window, min_periods=roll_decile_window)
        .apply(rolling_decile_func, raw=True)
    )
    return df


def build_position_from_decile(
    df_with_decile: pd.DataFrame,
    *,
    long_deciles: List[int],
    short_deciles: List[int],
    flat_deciles: List[int],
    abs_weight_map: Optional[Dict[int, float]] = None,
    decile_col: str = "decile",
    out_col: str = "position",
) -> pd.DataFrame:
    """
    在已经有 decile 的 df 上快速生成 position，不重复算 rolling。
    """
    df = df_with_decile.copy()

    # 只保留 decile 已经算出来的行
    df = df.dropna(subset=[decile_col]).reset_index(drop=True)
    df[decile_col] = df[decile_col].astype(int)

    long_set, short_set, flat_set = set(long_deciles), set(short_deciles), set(flat_deciles)
    if (long_set & short_set) or (long_set & flat_set) or (short_set & flat_set):
        raise ValueError("long/short/flat deciles 之间有重叠，请保证互不相交。")

    sign = np.zeros(len(df), dtype=int)
    sign[np.isin(df[decile_col].to_numpy(), list(long_set))] = 1
    sign[np.isin(df[decile_col].to_numpy(), list(short_set))] = -1
    sign[np.isin(df[decile_col].to_numpy(), list(flat_set))] = 0

    if abs_weight_map is None:
        abs_weight_map = {}

    abs_w = df[decile_col].map(lambda d: abs_weight_map.get(int(d), 1.0)).astype(float).to_numpy()
    pos = np.clip(sign * abs_w, -1.0, 1.0)
    df[out_col] = pos
    df["signal"] = sign
    return df


def sr_calculater(
    df: pd.DataFrame,
    time_col: str,
    factor_col: str,
    *,
    trade_price_col: str = "parkinson",
    roll_decile_window: int = 90,
    n_deciles: int = 10,
    oos_start: str = "2024-05-01",
    trade_lag: int = 1,
    use_zero_as_flat: bool = True,
) -> List[float]:
    """
    你的功能封装版：
    - 每个 i：只做多 decile=i（short为空）
    - 计算样本内 Sharpe 均值
    - 返回 sr_weight（长度 n_deciles）
    """
    # 1) decile 只算一次（关键提速点）
    df_dec = compute_rolling_decile(
        df=df,
        time_col=time_col,
        factor_col=factor_col,
        roll_decile_window=roll_decile_window,
        n_deciles=n_deciles,
    )

    sr_weight: List[float] = []
    abs_weight_map = {k: 1.0 for k in range(n_deciles)}  # 你原来的全 1

    for i in range(n_deciles):
        print(f"第{i}组")
        df_sig = build_position_from_decile(
            df_with_decile=df_dec,
            long_deciles=[i],
            short_deciles=[],
            flat_deciles=[],
            abs_weight_map=abs_weight_map,
            decile_col="decile",
            out_col="position",
        )

        trades_df, df_bar, yearly_stats = build_trades_from_signal_2(
            df=df_sig,
            time_col=time_col,
            price_col=trade_price_col,
            signal_col="position",
            plot=False,
            oos_start=oos_start,
            trade_lag=trade_lag,
            use_zero_as_flat=use_zero_as_flat,
        )

        sr = yearly_stats.loc[yearly_stats["Sample"] == "IS", "Sharpe"].mean()
        sr_weight.append(float(sr) if pd.notna(sr) else np.nan)
        print(f"样本内夏普均值：{sr}")
        print("--------------------------------")

        

    long_deciles  = tuple(i for i, sr in enumerate(sr_weight) if sr > 1)
    short_deciles = tuple(i for i, sr in enumerate(sr_weight) if sr < -1)
    print("long_deciles =", long_deciles)
    print("short_deciles =", short_deciles)

    sr_weight_abs = [abs(i) for i in sr_weight]
    abs_weight_map = {i: float(sigmoid_scale(sr_weight_abs)[i]) for i in range(len(sigmoid_scale(sr_weight_abs)))}
        
    return sr_weight,long_deciles,short_deciles,abs_weight_map


