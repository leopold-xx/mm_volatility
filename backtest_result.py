import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def build_yearly_stats_simple(
    df_daily: pd.DataFrame,
    initial_cash: float = 100000.0,
    initial_equity: float = 10000.0,
    oos_start: str = "2024-05-01",
    annual_days: int = 365,
) -> pd.DataFrame:
    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    oos_ts = pd.to_datetime(oos_start)

    # 单利日收益率（用于Sharpe）
    df["ret"] = df["daily_pnl"] / float(initial_cash)
    df["long_ret"] = df["daily_long_pnl"] / float(initial_cash)
    df["short_ret"] = df["daily_short_pnl"] / float(initial_cash)

    # 每日换手（估算）：成交次数 * 单次名义 / 初始资金
    df["daily_turnover"] = df["orders_filled"].astype(float) * float(initial_equity) / float(initial_cash)

    def sharpe(x: pd.Series) -> float:
        s = x.std(ddof=1)
        if s == 0 or np.isnan(s):
            return np.nan
        return (x.mean() / s) * np.sqrt(annual_days)

    def maxdd_pct_from_pnl(pnl_series: pd.Series) -> float:
        # 以该段开头为0，equity 从 initial_cash 起
        equity = float(initial_cash) + pnl_series.cumsum()
        peak = equity.cummax()
        dd_pct = equity / peak - 1.0
        return float(dd_pct.min())  # 负数

    rows = []
    for sample_name, sub in [("IS", df[df["date"] < oos_ts]), ("OOS", df[df["date"] >= oos_ts])]:
        if sub.empty:
            continue
        sub = sub.copy()
        sub["Year"] = sub["date"].dt.year

        for y, g in sub.groupby("Year"):
            pnl_sum = float(g["daily_pnl"].sum())
            long_sum = float(g["daily_long_pnl"].sum())
            short_sum = float(g["daily_short_pnl"].sum())

            ntrades = int(g["orders_filled"].sum())
            notional = float(ntrades) * float(initial_equity)

            pnl_per_trade = (pnl_sum / ntrades) if ntrades > 0 else np.nan
            bps_per_trade = (pnl_sum / notional * 1e4) if notional > 0 else np.nan

            rows.append({
                "Sample": sample_name,
                "Year": int(y),

                # 单利 return：年PnL / initial_cash
                "Return": pnl_sum / float(initial_cash),
                "LongReturn": long_sum / float(initial_cash),
                "ShortReturn": short_sum / float(initial_cash),

                "Sharpe": sharpe(g["ret"]),
                "MaxDD": maxdd_pct_from_pnl(g["daily_pnl"]),

                "NTrades": int(g["orders_filled"].sum()),
                "DailyTurnover": float(g["daily_turnover"].mean()),

                "PnLPerTrade": pnl_per_trade,        # USDT / fill
                "BpsPerTrade": bps_per_trade,        # bps on traded notional

            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Sample", "Year"]).reset_index(drop=True)
    return out


def plot_pnl_dd_with_table_simple(
    summary_df_final: pd.DataFrame,
    initial_cash: float = 100000.0,
    initial_equity: float = 10000.0,
    oos_start: str = "2024-05-01",
    annual_days: int = 365,
    price_mode: str = "mid",  # "mid"= (low+high)/2 ；你也可以改成别的列
    plot: bool = True,
):
    df = summary_df_final.copy()

    # 允许 date 是字符串 YYYY-MM-DD
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        time_col = "date"
    else:
        raise ValueError("summary_df_final 需要有 'date' 列（YYYY-MM-DD）")

    df = df.sort_values(time_col).reset_index(drop=True)

    # --- daily series（单利） ---
    df["daily_pnl"] = df["total_pnl"].astype(float)
    df["daily_long_pnl"] = df["long_close_profit"].astype(float)
    df["daily_short_pnl"] = df["short_close_profit"].astype(float)

    df["cum_pnl"] = df["daily_pnl"].cumsum()
    df["cum_long_pnl"] = df["daily_long_pnl"].cumsum()
    df["cum_short_pnl"] = df["daily_short_pnl"].cumsum()

    # equity & drawdown（单位 USDT）
    df["equity"] = float(initial_cash) + df["cum_pnl"]
    df["equity_peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] - df["equity_peak"]  # <=0, USDT

    # price series（用 daily mid）
    if price_mode == "mid":
        if ("price_low" in df.columns) and ("price_high" in df.columns):
            df["price_plot"] = (df["price_low"].astype(float) + df["price_high"].astype(float)) / 2.0
        else:
            df["price_plot"] = np.nan
    else:
        # 允许你传入某个列名
        if price_mode in df.columns:
            df["price_plot"] = df[price_mode].astype(float)
        else:
            df["price_plot"] = np.nan

    # yearly stats table (IS/OOS)
    yearly_stats = build_yearly_stats_simple(
        df_daily=df[[time_col, "daily_pnl", "daily_long_pnl", "daily_short_pnl", "orders_filled"]].assign(date=df[time_col]),
        initial_cash=initial_cash,
        initial_equity=initial_equity,
        oos_start=oos_start,
        annual_days=annual_days,
    )

    if not plot:
        return df, yearly_stats

    oos_start_ts = pd.to_datetime(oos_start)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.0, 1.6])

    # 顶部：总收益 + Drawdown
    ax_eq = fig.add_subplot(gs[0, 0])
    ax_eq.plot(df[time_col], df["cum_pnl"], label="Total PnL")
    ax_eq.set_ylabel("Cum PnL (simple, USDT)")
    ax_eq.set_title("Cumulative PnL & Drawdown (simple)")
    ax_eq.grid(alpha=0.3)

    ax_dd = ax_eq.twinx()
    ax_dd.fill_between(df[time_col], df["drawdown"], 0.0, alpha=0.6, label="Drawdown")
    ax_dd.set_ylabel("Drawdown (USDT)")

    lines1, labels1 = ax_eq.get_legend_handles_labels()
    lines2, labels2 = ax_dd.get_legend_handles_labels()
    ax_eq.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # 中间：多空 + 价格
    ax_ls = fig.add_subplot(gs[1, 0], sharex=ax_eq)
    ax_ls.plot(df[time_col], df["cum_long_pnl"], label="Long PnL")
    ax_ls.plot(df[time_col], df["cum_short_pnl"], label="Short PnL")
    ax_ls.plot(df[time_col], df["cum_pnl"], label="Total PnL", linewidth=1.2)
    ax_ls.set_ylabel("Cum PnL (USDT)")
    ax_ls.grid(alpha=0.3)

    ax_price = ax_ls.twinx()
    if df["price_plot"].notna().any():
        ax_price.plot(df[time_col], df["price_plot"], label="Price (mid)" if price_mode == "mid" else f"Price ({price_mode})", alpha=0.7)
        ax_price.set_ylabel("Price level")

    lines_ls, labels_ls = ax_ls.get_legend_handles_labels()
    lines_pr, labels_pr = ax_price.get_legend_handles_labels()
    ax_ls.legend(lines_ls + lines_pr, labels_ls + labels_pr, loc="upper left", fontsize=8)

    # X轴年份刻度
    if np.issubdtype(df[time_col].dtype, np.datetime64):
        for ax in [ax_eq, ax_ls]:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax_eq.get_xticklabels(), visible=False)

    ax_ls.set_xlabel("Time")

    # 样本外区域高亮
    if df[time_col].max() >= oos_start_ts:
        for ax in [ax_eq, ax_ls]:
            ax.axvspan(oos_start_ts, df[time_col].iloc[-1], alpha=0.12)
            ax.axvline(oos_start_ts, linestyle="--", linewidth=1)
        ax_eq.text(oos_start_ts, ax_eq.get_ylim()[1] * 0.9, "OOS", fontsize=9, ha="left", va="top")

    # 底部：年度统计表
    ax_table = fig.add_subplot(gs[2, 0])
    ax_table.axis("off")

    if not yearly_stats.empty:
        cols_show = ["Sample", "Year", "Return", "LongReturn", "ShortReturn", "Sharpe", "MaxDD", "NTrades", "DailyTurnover", "PnLPerTrade", "BpsPerTrade"]
        table_data = yearly_stats[cols_show].copy()

        # 格式化显示
        for c in ["Return", "LongReturn", "ShortReturn", "Sharpe", "MaxDD", "DailyTurnover","PnLPerTrade", "BpsPerTrade"]:
            table_data[c] = table_data[c].astype(float).round(4)

        table = ax_table.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.4)
        ax_table.set_title("Yearly stats (IS / OOS, simple)", fontsize=11, pad=18)

    fig.suptitle("Backtest (simple PnL)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
    plt.show()

    return df, yearly_stats