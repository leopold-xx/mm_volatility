# factor.py
from __future__ import annotations

import polars as pl


def factor_dpr(
    df: pl.DataFrame,
    *,
    ts_col: str = "ts_hour",
    price_col: str = "close",
    vol_col: str = "rv",
    factor_col_name: str = "factor",
    window: int = 90,
    short_window: int = 48,
    long_window: int = 168,
    price_mode: int = 1,
    distance_p: float = 2.0,
    vol_mode: int = 1,
    longshort_mode: int = 1,
    direction_mode: int = 1,
    eps: float = 1e-12,
) -> pl.DataFrame:
    """
    DPR 因子（chord_length / path_length）

    定义：
      chord_length = ||(ΣΔt, ΣΔP)||_p
      path_length  = Σ ||(Δt_i, ΔP_i)||_p
      DPR          = chord_length / (path_length + eps)

    其中 Δt 用 long_window 上 rolling_mean(|ΔP|) 来缩放，使时间尺度与价格尺度同量级。

    mode:
      price_mode:
        1: ΔP
        2: ΔlogP
        3: Δ(ΔP)
        4: |Δ(ΔP)|
        5: ΔP * log(|ΔP| + eps)   (修复 log(0) 问题)
      distance_p:
        1: L1
        else: p-norm
      vol_mode:
        1: 不变
        2: * vol_window_sum
        3: / vol_window_sum
      longshort_mode:
        1: 用 window 的 DPR
        2: DPR_long - DPR_short
        3: DPR_long / DPR_short
      direction_mode:
        1: 不变
        2: 乘方向比率 ΣΔP / (Σ|ΔP|+eps)
    """

    df = df.sort(ts_col)

    # ---------- 1. ΔP ----------
    price_f = pl.col(price_col).cast(pl.Float64)

    if price_mode == 1:
        dP_expr = price_f.diff()
    elif price_mode == 2:
        dP_expr = price_f.log().diff()
    elif price_mode == 3:
        dP_expr = price_f.diff().diff()
    elif price_mode == 4:
        dP_expr = price_f.diff().diff().abs()
    elif price_mode == 5:
        dP = price_f.diff()
        dP_expr = dP * (dP.abs() + eps).log()
    else:
        raise ValueError("price_mode 只能是 1/2/3/4/5")

    df = df.with_columns(dP_expr.alias("_dP"))

    # ---------- 1.5 计算 Δt 缩放项 ----------
    df = df.with_columns([
        pl.col("_dP").abs().alias("_abs_dP"),
        pl.col("_dP")
          .abs()
          .rolling_mean(window_size=long_window, min_samples=long_window)
          .alias("_dT"),
    ])

    # ---------- 2. 每段长度 seg_len ----------
    if distance_p == 1:
        seg_len_expr = pl.col("_abs_dP") + pl.col("_dT").abs()
    else:
        seg_len_expr = (
            pl.col("_abs_dP") ** distance_p
            + pl.col("_dT").abs() ** distance_p
        ) ** (1.0 / distance_p)

    df = df.with_columns(seg_len_expr.alias("_seg_len"))

    # ---------- 3. window 内 path/chord ----------
    df = df.with_columns([
        pl.col("_seg_len").rolling_sum(window_size=window, min_samples=window).alias("_path_len_w"),
        pl.col("_dP").rolling_sum(window_size=window, min_samples=window).alias("_sum_dP_w"),
        pl.col("_abs_dP").rolling_sum(window_size=window, min_samples=window).alias("_sum_abs_dP_w"),
        pl.col("_dT").rolling_sum(window_size=window, min_samples=window).alias("_sum_dT_w"),
    ])

    if distance_p == 1:
        chord_len_expr_w = pl.col("_sum_dP_w").abs() + pl.col("_sum_dT_w").abs()
    else:
        chord_len_expr_w = (
            pl.col("_sum_dP_w").abs() ** distance_p
            + pl.col("_sum_dT_w").abs() ** distance_p
        ) ** (1.0 / distance_p)

    df = df.with_columns(chord_len_expr_w.alias("_chord_len_w"))
    df = df.with_columns((pl.col("_chord_len_w") / (pl.col("_path_len_w") + eps)).alias("_DPR_w"))

    # ---------- 4. 长短线 DPR ----------
    df = df.with_columns([
        pl.col("_seg_len").rolling_sum(window_size=short_window, min_samples=short_window).alias("_path_len_s"),
        pl.col("_seg_len").rolling_sum(window_size=long_window,  min_samples=long_window).alias("_path_len_l"),
        pl.col("_dP").rolling_sum(window_size=short_window, min_samples=short_window).alias("_sum_dP_s"),
        pl.col("_dP").rolling_sum(window_size=long_window,  min_samples=long_window).alias("_sum_dP_l"),
        pl.col("_dT").rolling_sum(window_size=short_window, min_samples=short_window).alias("_sum_dT_s"),
        pl.col("_dT").rolling_sum(window_size=long_window,  min_samples=long_window).alias("_sum_dT_l"),
    ])

    if distance_p == 1:
        chord_s_expr = pl.col("_sum_dP_s").abs() + pl.col("_sum_dT_s").abs()
        chord_l_expr = pl.col("_sum_dP_l").abs() + pl.col("_sum_dT_l").abs()
    else:
        chord_s_expr = (
            pl.col("_sum_dP_s").abs() ** distance_p
            + pl.col("_sum_dT_s").abs() ** distance_p
        ) ** (1.0 / distance_p)
        chord_l_expr = (
            pl.col("_sum_dP_l").abs() ** distance_p
            + pl.col("_sum_dT_l").abs() ** distance_p
        ) ** (1.0 / distance_p)

    df = df.with_columns([
        (chord_s_expr / (pl.col("_path_len_s") + eps)).alias("_DPR_s"),
        (chord_l_expr / (pl.col("_path_len_l") + eps)).alias("_DPR_l"),
    ])

    # ---------- 5. 结合波动率 ----------
    df = df.with_columns(
        pl.col(vol_col).cast(pl.Float64)
          .rolling_sum(window_size=window, min_samples=window)
          .alias("_vol_w")
    )

    # ---------- 6. 方向信息 ----------
    df = df.with_columns(
        (pl.col("_sum_dP_w") / (pl.col("_sum_abs_dP_w") + eps)).alias("_dir_ratio")
    )

    # ---------- 7. 组合 mode ----------
    factor_expr = pl.col("_DPR_w")

    if longshort_mode == 2:
        factor_expr = pl.col("_DPR_l") - pl.col("_DPR_s")
    elif longshort_mode == 3:
        factor_expr = pl.col("_DPR_l") / (pl.col("_DPR_s") + eps)
    elif longshort_mode != 1:
        raise ValueError("longshort_mode 只能是 1/2/3")

    if vol_mode == 2:
        factor_expr = factor_expr * pl.col("_vol_w")
    elif vol_mode == 3:
        factor_expr = factor_expr / (pl.col("_vol_w") + eps)
    elif vol_mode != 1:
        raise ValueError("vol_mode 只能是 1/2/3")

    if direction_mode == 2:
        factor_expr = factor_expr * pl.col("_dir_ratio")
    elif direction_mode != 1:
        raise ValueError("direction_mode 只能是 1/2")

    df = df.with_columns(factor_expr.alias(factor_col_name))

    # ---------- 8. 清理中间列 ----------
    df = df.drop([
        "_dP", "_abs_dP", "_dT", "_seg_len",
        "_path_len_w", "_sum_dP_w", "_sum_abs_dP_w", "_sum_dT_w",
        "_chord_len_w", "_DPR_w",
        "_path_len_s", "_path_len_l", "_sum_dP_s", "_sum_dP_l",
        "_sum_dT_s", "_sum_dT_l", "_DPR_s", "_DPR_l",
        "_vol_w", "_dir_ratio",
    ], strict=False)

    return df

def factor_1(
    df: pl.DataFrame,
    window: int = 90,
    ts_col: str = "ts",
    vol_col: str = "rv",
    factor_col_name: str | None = None,
    cal_type: int | None = 1,
):
    """
    均线距离因子 = vol - rolling_mean(vol, window)
    """
    if factor_col_name is None:
        factor_col_name = f"factor"

    df = df.sort(ts_col)

    df = df.with_columns([
        pl.col(vol_col)
        .rolling_mean(
            window_size=window,
            min_samples=window,  # 至少有 window 个点再开始出均线
        )
        .alias(f"ma_{window}h")
    ])
    if cal_type == 1:
        df = df.with_columns([
            (pl.col(vol_col) - pl.col(f"ma_{window}h")).alias(factor_col_name)
        ])
    elif cal_type == 2:
        df = df.with_columns([
            (pl.col(vol_col) - pl.col(f"ma_{window}h")).abs().alias(factor_col_name)
        ])
    elif cal_type == 3:
        df = df.with_columns([
            (pl.col(vol_col) / (pl.col(f"ma_{window}h") + 1e-12) - 1.0)
            .alias(factor_col_name)
        ])

    return df

