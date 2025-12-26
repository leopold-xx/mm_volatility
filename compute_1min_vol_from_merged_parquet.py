#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm


# --------------------------
# kernels / vol functions
# --------------------------
def parzen_kernel(x: float) -> float:
    ax = abs(x)
    if ax <= 0.5:
        return 1 - 6 * (ax**2) + 6 * (ax**3)
    elif ax <= 1.0:
        return 2 * (1 - ax) ** 3
    else:
        return 0.0


def realized_kernel_parzen_from_prices(prices: np.ndarray, max_lag: int = 90) -> float:
    """
    更标准的 Realized Kernel (Parzen)（sum 自协方差形式）：
      r = diff(log(p))
      rk_var = sum(r^2) + 2 * sum_{h=1..H} k(h/(H+1)) * sum_{t=h+1..n} r_t r_{t-h}
      rk = sqrt(max(rk_var, 0))
    返回：rk（波动率口径）
    """
    p = np.asarray(prices, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size <= 1:
        return 0.0

    r = np.diff(np.log(p))
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return 0.0

    H = int(min(max_lag, n - 1))
    # gamma0 = sum r^2
    rk_var = float(np.dot(r, r))

    if H > 0:
        for h in range(1, H + 1):
            gamma_h = float(np.dot(r[h:], r[:-h]))   # sum autocov
            w = parzen_kernel(h / (H + 1))
            rk_var += 2.0 * w * gamma_h

    if rk_var < 0:
        rk_var = 0.0

    return float(np.sqrt(rk_var))


def rv_from_prices(prices: np.ndarray) -> float:
    """RV = sqrt(sum r^2), r = diff(log(p))"""
    p = np.asarray(prices, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size <= 1:
        return 0.0
    r = np.diff(np.log(p))
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    return float(np.sqrt(np.dot(r, r)))


def parkinson_from_high_low(high: float, low: float) -> float:
    """Parkinson sigma = sqrt( ln(H/L)^2 / (4 ln 2) )"""
    if (high is None) or (low is None) or (low <= 0) or (high <= 0) or (high < low):
        return 0.0
    x = np.log(high / low)
    var_p = (x * x) / (4.0 * np.log(2.0))
    return float(np.sqrt(var_p))


# --------------------------
# core
# --------------------------
def compute_1min_volatility_from_merged_parquet(
    input_dir: str,
    output_parquet: str,
    start_date: str = "2022-08-11",
    time_col: str = "last_transact_time",
    price_col: str = "close",
    max_lag: int = 90,
    skip_dot_underscore: bool = True,
):
    in_dir = Path(input_dir).expanduser().resolve()
    out_path = Path(output_parquet).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_ts = datetime.strptime(start_date, "%Y-%m-%d")

    if not in_dir.exists():
        raise FileNotFoundError(f"input_dir not exists: {in_dir}")

    files = sorted(in_dir.glob("*.parquet"))
    if skip_dot_underscore:
        files = [p for p in files if not p.name.startswith("._")]

    if not files:
        raise FileNotFoundError(f"no parquet files found in: {in_dir}")

    print(f"[INFO] input_dir  : {in_dir}")
    print(f"[INFO] files      : {len(files)}")
    print(f"[INFO] start_date : {start_date}")
    print(f"[INFO] bar_freq   : 1m")
    print(f"[INFO] price_col  : {price_col}")
    print(f"[INFO] output_parquet : {out_path}")
    print("")

    all_parts: list[pl.DataFrame] = []

    # need_cols = {time_col, "open", "high", "low", "close", price_col}

    for f in tqdm(files, desc="Processing parquet"):
        df = pl.read_parquet(str(f))

        # 时间列转 Datetime（如果是 Utf8）
        if df.schema[time_col] == pl.Utf8:
            df = df.with_columns(pl.col(time_col).str.to_datetime().alias(time_col))

        # 过滤日期
        df = df.filter(pl.col(time_col) >= pl.lit(start_ts))
        if df.is_empty():
            continue

        # 1) 排序 + 生成分钟桶
        df = df.sort(time_col).with_columns(
            pl.col(time_col).dt.truncate("1m").alias("ts_1m")
        )

        # 2) 1m 聚合：只保留你要的列
        #    聚合规则：
        #      open: first
        #      close: last
        #      high: max
        #      low: min
        #      quantity_sum: sum
        #      max_quantity: max
        #      min_quantity: min
        #      quantity_mean: mean
        #      is_buyer_maker: last（分钟末方向）
        #      vwap: 用 quantity_sum 做加权平均（更合理）
        #      price_mean: 也用 quantity_sum 做加权平均（更合理）
        #    同时收集 _prices 给 RV/RK 用（用 price_col，比如 close 或 vwap）
        bars = df.group_by("ts_1m").agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),

            pl.col("price_mean").cast(pl.Float64).alias("_price_mean_list"),
            pl.col("vwap").cast(pl.Float64).alias("_vwap_list"),
            pl.col("quantity_sum").cast(pl.Float64).alias("_qty_list"),

            pl.col("quantity_sum").cast(pl.Float64).sum().alias("quantity_sum"),
            pl.col("max_quantity").cast(pl.Float64).max().alias("max_quantity"),
            pl.col("min_quantity").cast(pl.Float64).min().alias("min_quantity"),
            pl.col("quantity_mean").cast(pl.Float64).mean().alias("quantity_mean"),

            pl.col("is_buyer_maker").last().alias("is_buyer_maker"),

            pl.col(price_col).cast(pl.Float64).alias("_prices"),
        ]).sort("ts_1m")

        bars = bars.with_columns([
            # vwap 加权：sum(vwap_i * qty_i) / sum(qty_i)
            pl.struct(["_vwap_list", "_qty_list"]).map_elements(
                lambda s: float(np.sum(np.array(s["_vwap_list"]) * np.array(s["_qty_list"])) / (np.sum(np.array(s["_qty_list"])) + 1e-12)),
                return_dtype=pl.Float64
            ).alias("vwap"),

            # price_mean 加权：sum(price_mean_i * qty_i) / sum(qty_i)
            pl.struct(["_price_mean_list", "_qty_list"]).map_elements(
                lambda s: float(np.sum(np.array(s["_price_mean_list"]) * np.array(s["_qty_list"])) / (np.sum(np.array(s["_qty_list"])) + 1e-12)),
                return_dtype=pl.Float64
            ).alias("price_mean"),
        ]).drop(["_vwap_list", "_price_mean_list", "_qty_list"])
    
        # 4) 计算 RV/RK/Parkinson
        bars = bars.with_columns([
            pl.col("_prices").map_elements(
                lambda x: rv_from_prices(np.array(x)) if len(x) >= 2 else 0.0,
                return_dtype=pl.Float64
            ).alias("rv"),
            pl.col("_prices").map_elements(
                lambda x: realized_kernel_parzen_from_prices(np.array(x), max_lag=max_lag) if len(x) >= 3 else 0.0,
                return_dtype=pl.Float64
            ).alias("rk"),
            pl.struct(["high", "low"]).map_elements(
                lambda s: parkinson_from_high_low(float(s["high"]), float(s["low"])),
                return_dtype=pl.Float64
            ).alias("parkinson"),
        ]).drop("_prices")

        # 5) 你要的表头：ts/date/hour/minute + 指定列
        bars = (
            bars
            .with_columns([
                pl.col("ts_1m").alias("ts"),
                pl.col("ts_1m").dt.date().alias("date"),
                pl.col("ts_1m").dt.hour().alias("hour"),
                pl.col("ts_1m").dt.minute().alias("minute"),
            ])
            .select([
                "ts","date","hour","minute",
                "open","high","low","close",
                "price_mean","quantity_sum","max_quantity","min_quantity","quantity_mean",
                "is_buyer_maker","vwap",
                "parkinson","rv","rk",
            ])
        )

        all_parts.append(bars)

    if not all_parts:
        print("[WARN] no data after filtering; nothing written.")
        return

    out = pl.concat(all_parts).unique(subset=["ts"]).sort("ts")
    out.write_parquet(str(out_path))
    print(f"[DONE] wrote: {out_path}  rows={out.height}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="merge 后 parquet 文件夹（*.parquet）")
    ap.add_argument("--output_parquet", required=True, help="输出 parquet 路径")
    ap.add_argument("--start_date", default="2022-08-11", help="YYYY-MM-DD (含)")
    ap.add_argument("--time_col", default="last_transact_time", help="时间列名（merge 后一般是 last_transact_time）")
    ap.add_argument("--price_col", default="close", help="用来算 RV/RK 的价格列（close 或 vwap）")
    ap.add_argument("--max_lag", type=int, default=90, help="RK 的最大滞后 H（分钟内会自动截断）")
    args = ap.parse_args()

    compute_1min_volatility_from_merged_parquet(
        input_dir=args.input_dir,
        output_parquet=args.output_parquet,
        start_date=args.start_date,
        time_col=args.time_col,
        price_col=args.price_col,
        max_lag=args.max_lag,
        skip_dot_underscore=True,
    )


if __name__ == "__main__":
    main()


# python3 code/compute_1min_vol_from_merged_parquet.py \
#   --input_dir "/Volumes/T7 Shield/data/binance_usd_arbusdt/aggtrades/parquet_merged" \
#   --output_parquet "/Volumes/T7 Shield/vol_1m/arb_vol_1m.parquet" \
#   --start_date "2022-08-11" \
#   --price_col "close" \
#   --max_lag 90