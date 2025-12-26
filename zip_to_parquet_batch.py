#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import sys
import zipfile
import argparse
from pathlib import Path
from datetime import datetime, date
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


DATE_RE = re.compile(r"-aggTrades-(\d{4}-\d{2}-\d{2})$")


def parse_date_from_zip(zip_path: Path) -> Optional[date]:
    """
    btcusdt-aggTrades-2022-08-11.zip -> 2022-08-11
    """
    stem = zip_path.stem  # 去掉 .zip
    m = DATE_RE.search(stem)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def convert_one_zip(
    zip_path_str: str,
    out_dir_str: str,
    overwrite: bool,
    compression: Optional[str],
    batch_size: int,
) -> Tuple[str, bool, str]:
    """
    返回: (zip_path, ok, message)
    """
    zip_path = Path(zip_path_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (zip_path.stem + ".parquet")
    if out_path.exists() and not overwrite:
        return (str(zip_path), True, "skip (exists): %s" % out_path.name)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            if not names:
                return (str(zip_path), False, "zip is empty")
            # Binance aggTrades 通常只有一个 CSV
            csv_name = names[0]

            with zf.open(csv_name, "r") as f:
                # 流式读 CSV（以 record batch 输出）
                read_opts = pacsv.ReadOptions(block_size=1 << 20)  # 1MB blocks
                parse_opts = pacsv.ParseOptions(delimiter=",")
                convert_opts = pacsv.ConvertOptions(strings_can_be_null=True)

                reader = pacsv.open_csv(
                    f,
                    read_options=read_opts,
                    parse_options=parse_opts,
                    convert_options=convert_opts,
                )

                writer = None
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

                try:
                    for batch in reader:
                        table = pa.Table.from_batches([batch])
                        if writer is None:
                            writer = pq.ParquetWriter(
                                tmp_path,
                                table.schema,
                                compression=compression,
                                use_dictionary=True,
                            )
                        writer.write_table(table, row_group_size=batch_size)

                    if writer is not None:
                        writer.close()
                    else:
                        return (str(zip_path), False, "no data batches")

                except Exception:
                    if writer is not None:
                        writer.close()
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except Exception:
                            pass
                    raise

        # 原子替换
        tmp_path.replace(out_path)
        return (str(zip_path), True, "ok -> %s" % out_path.name)

    except Exception as e:
        return (str(zip_path), False, "error: %s: %s" % (type(e).__name__, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg_dir", type=str, required=True, help="aggtrades目录，例如 ./binance_usd_btcusdt/aggtrades or /mnt/nas/cta_public/MarketData/BinanceSource/binance_usd_adausdt")
    ap.add_argument("--start_date", type=str, default="2020-06-01", help="YYYY-MM-DD (含)")
    ap.add_argument("--out_dir", type=str, default=None, help="parquet输出目录，默认在 agg_dir 下的 parquet/ 子目录")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已有 parquet")
    ap.add_argument("--compression", type=str, default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="并行进程数")
    ap.add_argument("--batch_size", type=int, default=1_000_000, help="Parquet row group size（影响写入速度/大小）")
    args = ap.parse_args()

    agg_dir = Path(args.agg_dir).resolve()
    if not agg_dir.exists():
        print("[FATAL] agg_dir not exists: %s" % agg_dir)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (agg_dir / "parquet")
    start_d = datetime.strptime(args.start_date, "%Y-%m-%d").date()

    zips = sorted(agg_dir.glob("*-aggTrades-*.zip"))
    targets = []
    for zp in zips:
        d = parse_date_from_zip(zp)
        if d is None:
            continue
        if d >= start_d:
            targets.append(zp)

    if not targets:
        print("[INFO] no zip files matched.")
        return

    compression = None if args.compression == "none" else args.compression

    print("[INFO] agg_dir  : %s" % agg_dir)
    print("[INFO] out_dir  : %s" % out_dir)
    print("[INFO] start    : %s" % start_d)
    print("[INFO] targets  : %d zip files" % len(targets))
    print("[INFO] workers  : %d" % args.workers)
    print("")

    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                convert_one_zip,
                str(zp),
                str(out_dir),
                args.overwrite,
                compression,
                args.batch_size,
            )
            for zp in targets
        ]

        for fut in as_completed(futs):
            zip_path, ok, msg = fut.result()
            if ok:
                if msg.startswith("skip"):
                    skip_cnt += 1
                else:
                    ok_cnt += 1
                print("[OK]  %s  %s" % (Path(zip_path).name, msg))
            else:
                fail_cnt += 1
                print("[BAD] %s  %s" % (Path(zip_path).name, msg))

    print("")
    print("[DONE] ok=%d, skip=%d, fail=%d" % (ok_cnt, skip_cnt, fail_cnt))


if __name__ == "__main__":
    main()

#     python3 zip_to_parquet_batch.py \
#   --agg_dir "/mnt/nas/cta_public/MarketData/BinanceSource/binance_usd_xrpusdt/aggtrades" \
#   --out_dir "/home/sixian/binance_usd_xrpusdt/aggtrades/parquet" \
#   --start_date "2020-06-01" \
#   --compression zstd \
#   --workers 32