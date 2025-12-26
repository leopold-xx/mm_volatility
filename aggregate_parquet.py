#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import argparse
from pathlib import Path
from datetime import datetime, date
from typing import Optional

import polars as pl
from tqdm import tqdm


# ============ 你的聚合函数（基本保持原样，只做少量健壮性增强） ============
REQUIRED_COLS = {
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
}

def convert_transact_time(df: pl.DataFrame) -> pl.DataFrame:
    if "transact_time" in df.columns:
        if df.schema["transact_time"] != pl.Datetime:
            df = df.with_columns(
                pl.from_epoch(pl.col("transact_time").cast(pl.Int64), time_unit="ms").alias("transact_time")
            )
    return df


def merge_trades(df: pl.DataFrame) -> pl.DataFrame:
    if df.schema.get("transact_time") != pl.Datetime:
        df = df.with_columns(
            pl.from_epoch(pl.col("transact_time").cast(pl.Int64), time_unit="ms").alias("transact_time")
        )

    df = df.with_columns([
        pl.col("price").cast(pl.Float64),
        pl.col("quantity").cast(pl.Float64),
        pl.col("agg_trade_id").cast(pl.Int64),
        pl.col("first_trade_id").cast(pl.Int64),
        pl.col("last_trade_id").cast(pl.Int64),
        pl.col("is_buyer_maker").cast(pl.Boolean, strict=False),
    ]).sort("transact_time")

    df = df.with_columns(
        (pl.col("is_buyer_maker") != pl.col("is_buyer_maker").shift(1))
        .fill_null(True)
        .alias("maker_changed")
    ).with_columns(
        pl.col("maker_changed").cast(pl.Int32).cum_sum().alias("group_id")
    )

    result = df.group_by("group_id").agg([
        pl.col("agg_trade_id").first().alias("first_agg_trade_id"),
        pl.col("agg_trade_id").last().alias("last_agg_trade_id"),

        (pl.col("price") * pl.col("quantity")).sum().alias("_total_value"),

        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").sort_by("transact_time").first().alias("open"),
        pl.col("price").sort_by("transact_time").last().alias("close"),
        pl.col("price").mean().alias("price_mean"),
        pl.col("price").std().fill_null(0).alias("price_std"),

        pl.col("quantity").sum().alias("quantity_sum"),
        pl.col("quantity").max().alias("max_quantity"),
        pl.col("quantity").min().alias("min_quantity"),
        pl.col("quantity").mean().alias("quantity_mean"),
        pl.col("quantity").std().fill_null(0).alias("quantity_std"),

        pl.col("first_trade_id").first().alias("first_trade_id"),
        pl.col("last_trade_id").last().alias("last_trade_id"),
        pl.col("transact_time").sort_by("transact_time").first().alias("first_transact_time"),
        pl.col("transact_time").sort_by("transact_time").last().alias("last_transact_time"),

        pl.col("is_buyer_maker").first().alias("is_buyer_maker"),

        pl.len().alias("trade_count"),
        (pl.col("price").sort_by("transact_time").diff().sign().diff().abs().sum() / 2).alias("price_turns"),
    ]).with_columns(
        pl.when(pl.col("quantity_sum") > 0)
          .then(pl.col("_total_value") / pl.col("quantity_sum"))
          .otherwise(pl.col("price_mean"))
          .alias("vwap")
    ).drop(["_total_value", "group_id"]).sort("first_agg_trade_id")

    return result


# ============ 批处理：遍历 parquet 文件夹，逐个聚合写出 ============

DATE_RE = re.compile(r"-aggTrades-(\d{4}-\d{2}-\d{2})$", re.IGNORECASE)

def parse_date_from_stem(stem: str) -> Optional[date]:
    m = DATE_RE.search(stem)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=str, help="/Volumes/T7 Shield/data/binance_usd_arbusdt/aggtrades/parquet")
    ap.add_argument("--output_dir", required=True, type=str, help="/Volumes/T7 Shield/data/binance_usd_arbusdt/aggtrades/parquet_merged")
    ap.add_argument("--start_date", default=None, type=str, help="可选：YYYY-MM-DD，只处理该日期(含)之后的文件（按文件名解析）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已有输出 parquet")
    ap.add_argument("--compression", default="zstd", type=str, choices=["zstd", "snappy", "gzip", "brotli", "lz4", "uncompressed"])
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"input_dir not exists: {in_dir}")

    start_d = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None

    files = sorted([p for p in in_dir.glob("*.parquet") if not p.name.startswith("._")])

    # 按文件名日期过滤（需要文件名包含 -aggTrades-YYYY-MM-DD）
    if start_d is not None:
        kept = []
        for p in files:
            d = parse_date_from_stem(p.stem)
            if d is None:
                continue
            if d >= start_d:
                kept.append(p)
        files = kept

    print(f"[INFO] input_dir : {in_dir}")
    print(f"[INFO] output_dir: {out_dir}")
    print(f"[INFO] files     : {len(files)}")
    if start_d:
        print(f"[INFO] start_date: {start_d}")
    print("")

    ok = skip = bad = 0

    for p in tqdm(files, desc="Aggregating"):
        out_path = out_dir / p.name  # 输出同名 parquet
        if out_path.exists() and not args.overwrite:
            skip += 1
            continue

        try:
            df = pl.read_parquet(str(p))
            if set(df.columns) != REQUIRED_COLS:
                bad += 1
                print(f"[SKIP] {p.name}: columns mismatch, got={df.columns}")
                continue
            merged = merge_trades(df)
            merged.write_parquet(str(out_path), compression=args.compression)
            ok += 1
        except Exception as e:
            bad += 1
            print(f"[BAD] {p.name}: {type(e).__name__}: {e}")

    print("")
    print(f"[DONE] ok={ok}, skip={skip}, bad={bad}")


if __name__ == "__main__":
    main()


# python3 code/aggregate_parquet.py \
#   --input_dir "/Volumes/T7 Shield/data/binance_usd_arbusdt/aggtrades/parquet" \
#   --output_dir "/Volumes/T7 Shield/data/binance_usd_arbusdt/aggtrades/parquet_merged"