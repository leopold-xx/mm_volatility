# factor_normalizer.py
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
import polars as pl


def winsorize_mad(series, n_mad: int = 5) -> np.ndarray:
    """MAD winsorization on a window."""
    x = np.asarray(series, dtype=float)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    if mad == 0 or np.isnan(mad):
        return x
    upper = median + n_mad * mad
    lower = median - n_mad * mad
    return np.clip(x, lower, upper)


def rolling_winsor_z(series: pd.Series, window: int, n_mad: int = 5) -> np.ndarray:
    """
    Rolling MAD winsor + z-score.
    返回与 series 等长的 numpy array，前 window-1 个为 NaN。
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)

    if n < window:
        return out

    for i in range(window - 1, n):
        w = values[i - window + 1 : i + 1]
        w2 = winsorize_mad(w, n_mad=n_mad)
        m = np.nanmean(w2)
        s = np.nanstd(w2, ddof=0)
        if (s == 0) or np.isnan(s):
            out[i] = 0.0
        else:
            out[i] = (w2[-1] - m) / (s + 1e-12)

    return out


def normalize_factor_2(
    df_pl: pl.DataFrame,
    vol_cols: List[str],
    factor_col: str,
    n_mad: int = 5,
    factor_roll_window: int = 90,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    输入: polars.DataFrame
    输出: pandas.DataFrame

    逻辑（与你原想法一致）：
    1) valid 行：factor 与所有 vol 非 NaN（这里不额外要求 vol>0；需要可以你再加）
    2) factor: rolling window 内做 MAD winsor -> zscore，得到 factor_col + "_clean"
    3) 合并回原 df；无效行保持 NaN
    4) 默认 dropna=True：丢掉含 NaN 的行（与你原代码一致）

    返回列：尽量保留常用字段 + vol_cols + factor + factor_clean（存在才保留）
    """
    df = df_pl.to_pandas()

    # valid mask
    valid_mask = ~df[factor_col].isna()
    for c in vol_cols:
        valid_mask &= ~df[c].isna()

    df_valid = df.loc[valid_mask].copy()

    clean_col = factor_col + "_clean"
    df_valid[clean_col] = rolling_winsor_z(
        df_valid[factor_col].astype(float),
        window=factor_roll_window,
        n_mad=n_mad,
    )

    # merge back
    df_out = df.copy()
    df_out.loc[valid_mask, clean_col] = df_valid[clean_col]

    if dropna:
        df_out = df_out.dropna().reset_index(drop=True)

    # 输出列（按你常用顺序）
    base_cols = ["ts", "ts_hour", "date", "hour", "minute", "open", "high", "low", "close"]
    keep_cols = [c for c in base_cols if c in df_out.columns]
    keep_cols += [c for c in vol_cols if c in df_out.columns]
    for c in [factor_col, clean_col]:
        if c in df_out.columns:
            keep_cols.append(c)

    # 去重保持顺序
    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

    return df_out[keep_cols]