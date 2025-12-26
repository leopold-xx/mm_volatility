# -*- coding: utf-8 -*-
"""
单品种单日动态波动率策略回测

用法：
    python demo_single_day_backtest_new.py

说明：
    - 使用动态波动率策略 (dynamic_volatility)
    - 单品种：BTCUSDT
    - 单日：2022-08-11（可修改）
    - 输出：回测摘要和PnL
"""

import sys
from pathlib import Path
import polars as pl
import pandas as pd
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mmcore_new.io.trades_loader import TradesLoader
from mmcore_new.metrics.calculator import IndicatorCalculator
from mmcore_new.sim.engine import BacktestEngine

# =================== 单日回测辅助函数 ===================
def attach_vol_regime(trades_df, vol_df, only_trade_in_vol_down=True):
    t = pl.from_pandas(trades_df).with_columns([
        pl.col("ts_exch").cast(pl.Int64),
        pl.col("price").cast(pl.Float64),
        pl.col("qty").cast(pl.Float64),
        pl.col("side").cast(pl.Utf8),
    ]).sort("ts_exch")

    vol_df = vol_df.copy()
    r = (
        pl.from_pandas(vol_df)
        .with_columns([
            pl.col("entry_time").cast(pl.Datetime).dt.epoch("ms").alias("entry_ms"),
            pl.col("exit_time").cast(pl.Datetime).dt.epoch("ms").alias("exit_ms"),
            (
                pl.col("size").cast(pl.Float64)
                * pl.when(pl.col("direction") == "long").then(1.0).otherwise(-1.0)
            ).alias("s_interval"),
        ])
        .select(["entry_ms", "exit_ms", "direction", "size", "s_interval"])
        .sort("entry_ms")
    )
    # backward asof: 找最近 entry
    j = t.join_asof(r, left_on="ts_exch", right_on="entry_ms", strategy="backward")
    # 只在 [entry, exit] 内有效，否则 s_t=0
    j = j.with_columns([
        pl.when(pl.col("exit_ms").is_not_null() & (pl.col("ts_exch") <= pl.col("exit_ms")))
          .then(pl.col("s_interval"))
          .otherwise(0.0)
          .alias("s_t")
    ])
    j = j.with_columns([
        (pl.col("s_t") > 0).alias("vol_up_now"),
        (pl.col("s_t") < 0).alias("vol_down_now"),
    ])
    if only_trade_in_vol_down:
        # 仅在 vol_down_now 才允许开仓做市；其他时候 risk_off
        j = j.with_columns([
            pl.col("vol_down_now").alias("trade_allowed"),
            (~pl.col("vol_down_now")).alias("risk_off"),
        ])
    else:
        # 不限制交易：仅在 up 时风险收缩
        j = j.with_columns([
            pl.lit(True).alias("trade_allowed"),
            (pl.col("vol_up_now")).alias("risk_off"),
        ])
    return j

def add_dynamic_controls(
    j: pl.DataFrame,
    enable_dynamic_sigma_multi=True,
    enable_dynamic_pos_punish=True,
    enable_dynamic_price_tolerance=True,
    # base
    sigma0=1.0, pos0=0.5, tol0=0.0002,
    # strength -> param
    alpha=0.5, beta=0.5, gamma=0.5,
    sigma_min=0.7, sigma_max=2.0,
    tol_min=1e-4, tol_max=5e-3,
):
    s = pl.col("s_t")  # long:+size, short:-size, neutral:0

    sigma_expr = (sigma0 * (1.0 + alpha * s)).clip(sigma_min, sigma_max)
    pos_expr   = (pos0 * (1.0 + beta * s)).clip(0.0, 1.0)
    tol_expr   = (tol0 * (1.0 + gamma * s)).clip(tol_min, tol_max)

    return j.with_columns([
        (sigma_expr if enable_dynamic_sigma_multi else pl.lit(sigma0)).alias("sigma_multi_t"),
        (pos_expr   if enable_dynamic_pos_punish else pl.lit(pos0)).alias("pos_punish_t"),
        (tol_expr   if enable_dynamic_price_tolerance else pl.lit(tol0)).alias("price_tol_t"),
    ])

# =================== 单日回测 ===================
def run_single_day_backtest(
    symbol: str = 'BTCUSDT',
    date: str = '2022-08-11',
    data_base_path: str = './',
    factor_df: Optional[pd.DataFrame] = None,
    use_dynamic_sigma_multi: bool = False,
    use_dynamic_pos_punish: bool = False,
    use_dynamic_price_tol: bool = True,
    use_risk_off: bool = True,
):
    """
    运行单品种单日动态波动率策略回测

    参数：
        symbol: 交易品种（如BTCUSDT）
        date: 回测日期（如2024-07-08）
        data_base_path: 数据根目录
    """
    if factor_df is None:
        factor_df = pd.read_csv("./vol_factor_trade_df.csv")

    print(f"\n{'='*60}")
    print(f"单品种单日回测 - 动态波动率策略")
    print(f"   品种: {symbol}")
    print(f"   日期: {date}")
    print(f"{'='*60}\n")

    # =================== 1. 构建数据路径 ===================
    folder_name = f"binance_usd_{symbol.lower()}"
    data_file = Path(data_base_path) / folder_name / "aggtrades" / "parquet" / f"{symbol}-aggTrades-{date}.parquet"

    print(f"📁 数据文件: {data_file}")

    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        print(f"   请确认数据路径或修改 data_base_path 参数")
        return None

    # =================== 2. 加载trades数据 ===================
    print(f"\n加载trades数据...")
    loader = TradesLoader()
    df, ticksize = loader.load_from_parquet(str(data_file), symbol)

    if df is None:
        print("数据加载失败")
        return None

    print(f"数据加载成功: {len(df):,} 条trades, ticksize={ticksize}")

    # =================== 2.1 加载vol数据 ===================
    vol_df = factor_df

    # =================== 2.2 拼接trade和vol ===================
    joined_df = attach_vol_regime(df, vol_df,only_trade_in_vol_down=True)

    # =================== 2.3 计算动态指标 ===================
    j_final = add_dynamic_controls(
        j=joined_df,
        enable_dynamic_sigma_multi=True,
        enable_dynamic_pos_punish=True,
        enable_dynamic_price_tolerance=True,
        sigma0=1.0, pos0=0.5, tol0=0.0002,
        alpha=0.5, beta=0.5, gamma=0.5,
        sigma_min=0, sigma_max=2.0,
        tol_min=1e-4, tol_max=5e-3,
    )
    # =================== 3. 计算技术指标 ===================
    print(f"\n计算技术指标...")

    # 动态波动率策略参数
    calculator = IndicatorCalculator(
        tau_p='5min',           # PEMA时间常数（公允价格响应速度）
        tau_o='1h',             # 波动率时间常数
        sigma_multi=1.0,        # 波动率缩放因子（挂单距离 = sigma * sigma_multi）
        min_volatility=0.001,   # 波动率下限（千分之一，防止价差过窄）
        guard_k=2               # Guard窗口大小（风控参数）
    )

    # 计算指标（包含pema、sigma、ask_permit、bid_permit、guard_ask、guard_bid）
    df_with_indicators = calculator.calculate_indicators(j_final.to_pandas(), symbol=symbol)

    print(f"指标计算完成")

    # =================== 4. 运行回测 ===================
    print(f"\n启动回测引擎...")

    engine = BacktestEngine(
        initial_cash=100000.0,      # 初始现金
        initial_equity=10000.0,     # 单次建仓权益（每笔订单用10000元）
        pos_punish=0.5,             # 仓位惩罚因子（持仓时订单向pema靠拢）
        strategy_config={
            'price_tolerance': 0.0002  # 价格容差（超出则撤单重挂）
        },
        output_mode='summary',      # 输出模式：summary聚合 或 detail逐笔
        aggregation_seconds=600,     # 聚合周期：10分钟

        ticksize= ticksize,
        sigma_multi= 1.0,
        use_dynamic_sigma_multi= use_dynamic_sigma_multi,
        use_dynamic_pos_punish= use_dynamic_pos_punish,
        use_dynamic_price_tol= use_dynamic_price_tol,
        use_risk_off=use_risk_off,
    )

    report = engine.run_backtest(df_with_indicators)

    print(f"回测完成")

    # =================== 5. 提取结果DataFrame ===================
    # run_backtest返回的是report字典，实际数据在engine.trade_history中
    if engine.trade_history is None or len(engine.trade_history) == 0:
        print("没有交易历史数据")
        return None

    # summary模式下的列名定义
    summary_columns = [
        'timestamp', 'datetime', 'open', 'high', 'low', 'close',
        'volume', 'turnover', 'buy_count', 'sell_count',
        'cash', 'position', 'total_value', 'bar_pnl', 'cumulative_pnl',
        'pema', 'ask_permit', 'bid_permit',
        'orders_placed', 'orders_filled', 'buy_orders', 'sell_orders',
        'buy_fills', 'sell_fills', 'filled_volume', 'filled_turnover',
        'long_close_profit', 'short_close_profit'
    ]

    # 转换为DataFrame
    result = pd.DataFrame(engine.trade_history, columns=summary_columns)
    result['datetime'] = pd.to_datetime(result['timestamp'], unit='ms')

    # =================== 6. 输出结果摘要 ===================
    print(f"\n{'='*60}")
    print(f"回测结果摘要")
    print(f"{'='*60}")

    # 基础统计
    total_pnl = result['cumulative_pnl'].iloc[-1]
    total_orders_placed = result['orders_placed'].sum()
    total_orders_filled = result['orders_filled'].sum()
    fill_rate = total_orders_filled / total_orders_placed * 100 if total_orders_placed > 0 else 0

    print(f"   累计PnL: {total_pnl:.2f} USDT")
    print(f"   下单次数: {int(total_orders_placed):,}")
    print(f"   成交次数: {int(total_orders_filled):,}")
    print(f"   成交率: {fill_rate:.1f}%")

    # 盈亏分解
    long_profit = result['long_close_profit'].sum()
    short_profit = result['short_close_profit'].sum()
    print(f"   多头平仓盈亏: {long_profit:.2f}")
    print(f"   空头平仓盈亏: {short_profit:.2f}")

    # 价格和成交量
    price_range = f"{result['low'].min():.2f} - {result['high'].max():.2f}"
    total_volume = result['volume'].sum()
    price_low = float(result['low'].min())
    price_high = float(result['high'].max())
    total_volume = float(result['volume'].sum())
    print(f"   价格范围: {price_range}")
    print(f"   总成交量: {total_volume:,.2f}")

    # 时间范围
    start_time = result['datetime'].iloc[0]
    end_time = result['datetime'].iloc[-1]
    print(f"   时间范围: {start_time} ~ {end_time}")
    summary_df = pd.DataFrame([{
        "symbol": symbol,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "ticksize": float(ticksize),

        "total_pnl": total_pnl,
        "long_close_profit": long_profit,
        "short_close_profit": short_profit,

        "orders_placed": int(total_orders_placed),
        "orders_filled": int(total_orders_filled),
        "fill_rate_pct": float(fill_rate),

        "price_low": price_low,
        "price_high": price_high,
        "total_volume": total_volume,

        "n_trades": int(len(df)),               # 当天 trades 行数（pandas df）
        "n_bars": int(len(result)),             # summary bar 数
    }])
    print("\n回测摘要（1行 dataframe）")
    print(summary_df)
    print(f"\n{'='*60}")
    print(f"回测完成！")
    print(f"{'='*60}\n")

    return result, summary_df

# =================== 多日回测辅助函数 ===================
def _build_date_list(start_date: str, end_date: str):
    """生成 [start_date, end_date] 的日期列表YYYY-MM-DD。"""
    dr = pd.date_range(start=start_date, end=end_date, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dr]

# =================== 并行化辅助函数 ===================
_FACTOR_DF = None
_FACTOR_PATH = None

def _init_worker(factor_df_path: str):
    global _FACTOR_DF, _FACTOR_PATH
    _FACTOR_PATH = factor_df_path
    df = pd.read_parquet(factor_df_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"]  = pd.to_datetime(df["exit_time"])
    _FACTOR_DF = df

def _worker_one_day(date: str, symbol: str, data_base_path: str, engine_flags: dict):
    global _FACTOR_DF
    try:
        out = run_single_day_backtest(
            symbol=symbol,
            date=date,
            data_base_path=data_base_path,
            factor_df=_FACTOR_DF,
            **engine_flags,  # NEW
        )
        if out is None:
            return None
        _, summary_df = out
        return summary_df
    except Exception:
        print(f"[ERROR] {symbol} {date}")
        print(traceback.format_exc())
        return None
        
# =================== 并行化===================
def run_multi_day_backtest_parallel(
    start_date: str,
    end_date: str,
    symbol: str = "BTCUSDT",
    data_base_path: str = "./",
    factor_df_path: str = "./vol_factor_trade_df.csv",
    max_workers: int = 4,
    # NEW: engine flags
    use_dynamic_sigma_multi: bool = False,
    use_dynamic_pos_punish: bool = False,
    use_dynamic_price_tol: bool = False,
    use_risk_off: bool = False,
):
    date_list = _build_date_list(start_date, end_date)

    engine_flags = dict(
        use_dynamic_sigma_multi=use_dynamic_sigma_multi,
        use_dynamic_pos_punish=use_dynamic_pos_punish,
        use_dynamic_price_tol=use_dynamic_price_tol,
        use_risk_off=use_risk_off,
    )

    outs = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(factor_df_path,),
    ) as ex:
        futures = {
            ex.submit(_worker_one_day, d, symbol, data_base_path, engine_flags): d
            for d in date_list
        }

        for fu in as_completed(futures):
            d = futures[fu]
            summary_df = fu.result()
            if summary_df is None:
                print(f"[SKIP] {symbol} {d}")
                continue
            outs.append(summary_df)

    if not outs:
        return pd.DataFrame()

    return (
        pd.concat(outs, ignore_index=True)
          .sort_values(["symbol", "date"])
          .reset_index(drop=True)
    )










if __name__ == "__main__":
    import time
    t0 = time.perf_counter()

    symbol_lower = 'fil'
    symbol = "FIL"
    factor_name = 'dpr_factor_2_3_3_1_1'  
    use_risk_off = False

    start_date = "2022-08-11"
    end_date = "2025-12-01"
    factor_df_path = f"/Volumes/T7 Shield/vol_factor_1m/{symbol_lower}/{factor_name}/{factor_name}_vol.parquet"
    
    
    summary_df_final = run_multi_day_backtest_parallel(
        start_date=start_date,
        end_date=end_date,
        symbol=f"{symbol}USDT",
        data_base_path="/Volumes/T7 Shield/data/",
        factor_df_path=factor_df_path,
        max_workers=6,

        # 这里随便调
        use_dynamic_sigma_multi=False,
        use_dynamic_pos_punish=False,
        use_dynamic_price_tol=False,
        use_risk_off=use_risk_off,
    )

    if use_risk_off:
        mode = "rr"
    else:
        mode = "basic"

    out_dir = Path(f"/Volumes/T7 Shield/vol_factor_1m/{symbol_lower}/{factor_name}")
    out_dir.mkdir(exist_ok=True)
    summary_df_final.to_parquet(out_dir / f"{mode}_{start_date}_{end_date}_{symbol}USDT_result.parquet", index=False)

    t1 = time.perf_counter()
    print(f"Total elapsed: {t1 - t0:.3f} s")
    print(f"已保存到{mode}_{start_date}_{end_date}_{symbol}USDT_result.parquet")