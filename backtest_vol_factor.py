import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def add_rolling_decile_signal_2(
    df: pd.DataFrame,
    time_col: str,
    factor_col: str,
    roll_decile_window: int = 90,
    n_deciles: int = 10,
    # ---- 策略模式：decile -> signal(-1/0/1) ----
    long_deciles=(),
    short_deciles=(),
    flat_deciles=(),
    # ---- 仓位权重：decile -> abs_weight (0~1) ----
    abs_weight_map: dict | None = None,
):
    """
    根据因子 rolling 分位生成:
        - decile: 0 ~ n_deciles-1
        - signal: -1 / 0 / 1 （方向信号，可由参数指定哪些 decile 映射到哪种方向）
        - position: signal * abs_weight(decile) ∈ [-1,1]

    参数
    ----
    long_deciles : iterable[int]
        这些 decile 映射到 signal = +1
    short_deciles : iterable[int]
        这些 decile 映射到 signal = -1
    flat_deciles : iterable[int]
        这些 decile 映射到 signal = 0（可选，不写就默认其它未落在 long/short 的 decile 也为 0）
        要求：三组 decile 互不相交（单射）
    abs_weight_map : dict[int, float] or None
        给每个 decile 一个绝对仓位权重 |w|，最后 position = signal * |w|
        如果为 None，默认使用一个例子权重（仅在 n_deciles==10 时给默认），
        其它 decile 若不在 map 中则默认 abs_weight = 1.0
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # ===== 1. 计算 rolling 分位 decile =====
    q_edges = np.linspace(0.0, 1.0, n_deciles + 1)

    def rolling_decile_func(window_vals: np.ndarray) -> float:
        current = window_vals[-1]
        qs = np.quantile(window_vals, q_edges)
        d = np.searchsorted(qs, current, side="right") - 1
        d = max(0, min(n_deciles - 1, d))
        return d

    df["decile"] = (
        df[factor_col]
        .rolling(window=roll_decile_window, min_periods=roll_decile_window)
        .apply(rolling_decile_func, raw=True)
    )

    # 前 window-1 没有 decile，直接丢掉
    df = df.dropna(subset=["decile"]).reset_index(drop=True)
    df["decile"] = df["decile"].astype(int)

    # ===== 2. 由参数生成 signal (-1/0/1) =====
    long_set = set(long_deciles)
    short_set = set(short_deciles)
    flat_set = set(flat_deciles)

    # 简单一致性检查：三组不能重叠
    if (long_set & short_set) or (long_set & flat_set) or (short_set & flat_set):
        raise ValueError("long_deciles / short_deciles / flat_deciles 之间有重叠，请保证三者互不相交。")

    sign = np.zeros(len(df), dtype=int)

    sign[np.isin(df["decile"], list(long_set))] = 1
    sign[np.isin(df["decile"], list(short_set))] = -1
    # flat_deciles 明确指定为 0，其余没指定的 decile 也默认是 0
    sign[np.isin(df["decile"], list(flat_set))] = 0

    df["signal"] = sign

    # ===== 3. 仓位权重：abs_weight_map(decile) → abs_w，position = sign * abs_w =====
    if abs_weight_map is None:
        # 如果用户没给，就在 n_deciles == 10 时用你的示例权重
        if n_deciles == 10:
            abs_weight_map = {
                0: 1.0,
                1: 0.9,
                2: 0.6,
                3: 0.2,
                4: 0.2,
                5: 0.2,
                6: 0.2,
                7: 0.6,
                8: 0.9,
                9: 1.0,
            }
        else:
            # 其它情况，默认所有 decile 使用 1.0 仓位
            abs_weight_map = {}

    # map 得到 abs_weight，没在 abs_weight_map 里的 decile 默认 1.0
    abs_w = df["decile"].map(lambda d: abs_weight_map.get(d, 1.0)).astype(float).to_numpy()

    # 仓位 = 方向 * 绝对权重，保证在 [-1,1]
    position = sign * abs_w
    position = np.clip(position, -1.0, 1.0)
    df["position"] = position

    return df




def build_trades_from_signal_2(
    df: pd.DataFrame,
    time_col: str,
    price_col: str,
    signal_col: str = "position",
    plot: bool = True,
    oos_start: str = "2024-05-01",   # 样本外起始日期
    trade_lag: int = 1,              # t 时间出信号, 用 t+trade_lag 的价格成交
    use_zero_as_flat: bool = True,   # 新增：False 时忽略 0 信号，只在正负交换时交易
):
    """
    分批建仓版本（单利 PnL）：

    - signal_t ∈ [-1,1] 被理解为“本 bar 想新增多少仓位”（增量），而不是目标仓位；
    - 每次新增仓位都记录自己的 entry_price 和 weight（size）；
    - 平仓时，每一条腿单独记一条 trade，size = 该腿的仓位权重；
    - trade_lag: 第 i 行的 signal_i 决定在第 i+trade_lag 行的价格上开仓/平仓；

    - use_zero_as_flat:
        * True  : signal == 0 表示想平仓 / 空仓（当前逻辑）
        * False : 完全忽略 0 信号，不平仓、不加仓，只在 signal 从 >0 变 <0
                  或从 <0 变 >0 时触发平仓/反手。
    """

    if trade_lag < 1:
        raise ValueError("trade_lag 必须 >= 1")

    # ===== 0. 排序 & 基础数据 =====
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    times = df[time_col].to_numpy()
    prices = df[price_col].astype(float).to_numpy()
    signals = np.clip(df[signal_col].astype(float).to_numpy(), -1.0, 1.0)

    N = len(df)

    # ===== 1. 逐 bar PnL（决策在 i，成交在 i+trade_lag） =====
    pnl_bar = np.zeros(N)
    long_pnl_bar = np.zeros(N)
    short_pnl_bar = np.zeros(N)

    trades = []

    current_dir = 0      # 当前方向：0/1/-1
    leg_weights = []     # 每一条腿的正仓位 (0~1)
    leg_prices = []      # 每一条腿的 entry price
    leg_entry_times = [] # 每一条腿的 entry_time
    leg_entry_indices = []  # 每一条腿的 entry index（bar 序号）

    max_start = N - trade_lag
    for i in range(max_start):
        s = signals[i]
        j = i + trade_lag        # 成交所在 bar
        px_exec = prices[j]
        t_exec = times[j]

        # ===== 关键改动：如何解释 signal == 0 =====
        if use_zero_as_flat:
            # 原逻辑：0 表示「想空仓」
            if s == 0.0:
                desired_dir = 0
            else:
                desired_dir = 1 if s > 0 else -1
        else:
            # 新逻辑：忽略 0，不触发平仓也不加仓
            #   * 0 时保持当前方向（不平）
            #   * 只有 s>0 / s<0 时才表示想做多 / 想做空
            if s == 0.0:
                desired_dir = current_dir   # 持有原方向
            else:
                desired_dir = 1 if s > 0 else -1

        total_w = sum(leg_weights)

        # ========== 情况 1：当前无仓位 ==========
        if current_dir == 0:
            if desired_dir != 0:
                add_w = min(abs(s), 1.0)
                if add_w > 0:
                    current_dir = desired_dir
                    leg_weights = [add_w]
                    leg_prices = [px_exec]
                    leg_entry_times = [t_exec]
                    leg_entry_indices = [j]

        # ========== 情况 2：当前是多头 ==========
        elif current_dir == 1:
            if desired_dir <= 0:
                # ---- 平掉所有多头，每一条腿单独一笔 trade ----
                for w, ep, et, eidx in zip(
                    leg_weights, leg_prices, leg_entry_times, leg_entry_indices
                ):
                    pnl = w * (px_exec - ep)
                    holding_bars = j - eidx

                    pnl_bar[j] += pnl
                    long_pnl_bar[j] += pnl

                    trades.append(
                        dict(
                            direction="long",
                            size=w,
                            entry_time=et,
                            exit_time=t_exec,
                            entry_price=ep,
                            exit_price=px_exec,
                            pnl=pnl,
                            holding_bars=holding_bars,
                        )
                    )

                # 清空
                current_dir = 0
                leg_weights, leg_prices = [], []
                leg_entry_times, leg_entry_indices = [], []

                # 反手做空（只有 desired_dir == -1 时）
                if desired_dir == -1 and abs(s) > 0:
                    add_w = min(abs(s), 1.0)
                    if add_w > 0:
                        current_dir = -1
                        leg_weights = [add_w]
                        leg_prices = [px_exec]
                        leg_entry_times = [t_exec]
                        leg_entry_indices = [j]

            else:
                # ---- 继续加多头仓位（分批建仓） ----
                remaining = max(0.0, 1.0 - total_w)
                add_w = min(abs(s), remaining)
                if add_w > 0:
                    leg_weights.append(add_w)
                    leg_prices.append(px_exec)
                    leg_entry_times.append(t_exec)
                    leg_entry_indices.append(j)

        # ========== 情况 3：当前是空头 ==========
        elif current_dir == -1:
            if desired_dir >= 0:
                # ---- 平掉所有空头，每一条腿单独一笔 trade ----
                for w, ep, et, eidx in zip(
                    leg_weights, leg_prices, leg_entry_times, leg_entry_indices
                ):
                    pnl = w * (ep - px_exec)
                    holding_bars = j - eidx

                    pnl_bar[j] += pnl
                    short_pnl_bar[j] += pnl

                    trades.append(
                        dict(
                            direction="short",
                            size=w,
                            entry_time=et,
                            exit_time=t_exec,
                            entry_price=ep,
                            exit_price=px_exec,
                            pnl=pnl,
                            holding_bars=holding_bars,
                        )
                    )

                # 清空
                current_dir = 0
                leg_weights, leg_prices = [], []
                leg_entry_times, leg_entry_indices = [], []

                # 反手做多
                if desired_dir == 1 and abs(s) > 0:
                    add_w = min(abs(s), 1.0)
                    if add_w > 0:
                        current_dir = 1
                        leg_weights = [add_w]
                        leg_prices = [px_exec]
                        leg_entry_times = [t_exec]
                        leg_entry_indices = [j]

            else:
                # ---- 继续加空头仓位（分批建仓） ----
                remaining = max(0.0, 1.0 - total_w)
                add_w = min(abs(s), remaining)
                if add_w > 0:
                    leg_weights.append(add_w)
                    leg_prices.append(px_exec)
                    leg_entry_times.append(t_exec)
                    leg_entry_indices.append(j)

    # ===== 2. 收尾：最后一笔强制平仓 =====
    if current_dir != 0 and len(leg_weights) > 0:
        px_last = prices[-1]
        t_last = times[-1]
        j = N - 1

        if current_dir == 1:
            for w, ep, et, eidx in zip(
                leg_weights, leg_prices, leg_entry_times, leg_entry_indices
            ):
                pnl = w * (px_last - ep)
                holding_bars = j - eidx
                pnl_bar[j] += pnl
                long_pnl_bar[j] += pnl

                trades.append(
                    dict(
                        direction="long",
                        size=w,
                        entry_time=et,
                        exit_time=t_last,
                        entry_price=ep,
                        exit_price=px_last,
                        pnl=pnl,
                        holding_bars=holding_bars,
                    )
                )
        else:
            for w, ep, et, eidx in zip(
                leg_weights, leg_prices, leg_entry_times, leg_entry_indices
            ):
                pnl = w * (ep - px_last)
                holding_bars = j - eidx
                pnl_bar[j] += pnl
                short_pnl_bar[j] += pnl

                trades.append(
                    dict(
                        direction="short",
                        size=w,
                        entry_time=et,
                        exit_time=t_last,
                        entry_price=ep,
                        exit_price=px_last,
                        pnl=pnl,
                        holding_bars=holding_bars,
                    )
                )

    trades_df = pd.DataFrame(trades)

    # ===== 3. 构建 df_bar & drawdown =====
    df_bar = df.copy()
    df_bar["pnl"] = pnl_bar
    df_bar["long_pnl"] = long_pnl_bar
    df_bar["short_pnl"] = short_pnl_bar
    df_bar["cum_pnl"] = df_bar["pnl"].cumsum()
    df_bar["cum_long_pnl"] = df_bar["long_pnl"].cumsum()
    df_bar["cum_short_pnl"] = df_bar["short_pnl"].cumsum()

    cum_max = df_bar["cum_pnl"].cummax()
    df_bar["drawdown"] = df_bar["cum_pnl"] - cum_max

    # ===== 4. 样本内 / 样本外划分 =====
    oos_start_ts = pd.Timestamp(oos_start)
    df_bar["Sample"] = np.where(df_bar[time_col] < oos_start_ts, "IS", "OOS")
    df_bar["Year"] = df_bar[time_col].dt.year

    if not trades_df.empty:
        trades_df["Sample"] = np.where(trades_df["entry_time"] < oos_start_ts, "IS", "OOS")
        trades_df["Year"] = trades_df["entry_time"].dt.year

    # ===== 5. 年度统计（含 NTrades / AvgHoldBars / DailyTurnover / BIPS） =====
    yearly_rows = []
    for (sample, year), g in df_bar.groupby(["Sample", "Year"]):
        pnl_year = g["pnl"]
        if len(pnl_year) == 0:
            continue

        long_pnl_year = g["long_pnl"].sum()
        short_pnl_year = g["short_pnl"].sum()
        total_return_year = pnl_year.sum()

        mean_r = pnl_year.mean()
        std_r = pnl_year.std(ddof=1)
        sharpe = np.nan if (std_r == 0 or np.isnan(std_r)) else mean_r / std_r * np.sqrt(len(pnl_year))

        cum_year = pnl_year.cumsum()
        cum_max_year = cum_year.cummax()
        dd_year = cum_year - cum_max_year
        max_dd_year = dd_year.min()

        if not trades_df.empty:
            trades_sub = trades_df[(trades_df["Sample"] == sample) & (trades_df["Year"] == year)]
            n_trades = len(trades_sub)
            avg_hold_bars = trades_sub["holding_bars"].mean() if n_trades > 0 else np.nan
            total_notional = trades_sub["size"].abs().sum()
            bips = np.nan
            if total_notional > 0:
                pnl_sum = trades_sub["pnl"].sum()
                bips = pnl_sum * 10000.0 / total_notional
        else:
            trades_sub = pd.DataFrame()
            n_trades = 0
            avg_hold_bars = np.nan
            total_notional = 0.0
            bips = np.nan

        # 日换手率：年内成交笔数 / 年内自然日数
        g_days = g[time_col].dt.normalize().nunique()
        daily_turnover = n_trades / g_days if g_days > 0 else np.nan

        yearly_rows.append(
            dict(
                Sample=sample,
                Year=year,
                Return=total_return_year,
                LongReturn=long_pnl_year,
                ShortReturn=short_pnl_year,
                Sharpe=sharpe,
                MaxDD=max_dd_year,
                NTrades=n_trades,
                AvgHoldBars=avg_hold_bars,
                DailyTurnover=daily_turnover,
                BIPS=bips,
            )
        )

    yearly_stats = pd.DataFrame(yearly_rows)
    if not yearly_stats.empty:
        sample_order = {"IS": 0, "OOS": 1}
        yearly_stats["SampleOrder"] = yearly_stats["Sample"].map(sample_order)
        yearly_stats = (
            yearly_stats.sort_values(["SampleOrder", "Year"])
            .drop(columns=["SampleOrder"])
            .reset_index(drop=True)
        )

    # ===== 6. 画图 =====
    if plot:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.0, 1.6])

        # 顶部：总收益 + Drawdown
        ax_eq = fig.add_subplot(gs[0, 0])
        ax_eq.plot(df_bar[time_col], df_bar["cum_pnl"], color="black", label="Total PnL")
        ax_eq.set_ylabel("Cum PnL (simple)")
        ax_eq.set_title(f"Cumulative PnL & Drawdown (trade_lag={trade_lag})")
        ax_eq.grid(alpha=0.3)

        ax_dd = ax_eq.twinx()
        ax_dd.fill_between(
            df_bar[time_col],
            df_bar["drawdown"],
            0.0,
            color="skyblue",
            alpha=0.6,
            label="Drawdown",
        )
        ax_dd.set_ylabel("Drawdown")

        lines1, labels1 = ax_eq.get_legend_handles_labels()
        lines2, labels2 = ax_dd.get_legend_handles_labels()
        ax_eq.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # 中间：多空 + 价格
        ax_ls = fig.add_subplot(gs[1, 0], sharex=ax_eq)
        ax_ls.plot(df_bar[time_col], df_bar["cum_long_pnl"], color="red", label="Long PnL")
        ax_ls.plot(df_bar[time_col], df_bar["cum_short_pnl"], color="green", label="Short PnL")
        ax_ls.plot(df_bar[time_col], df_bar["cum_pnl"], color="black", label="Total PnL", linewidth=1.2)
        ax_ls.set_ylabel("Cum PnL")
        ax_ls.grid(alpha=0.3)

        ax_price = ax_ls.twinx()
        ax_price.plot(df_bar[time_col], df_bar[price_col], color="gold", label="Vol", alpha=0.7)
        ax_price.set_ylabel("Vol level")

        lines_ls, labels_ls = ax_ls.get_legend_handles_labels()
        lines_pr, labels_pr = ax_price.get_legend_handles_labels()
        ax_ls.legend(lines_ls + lines_pr, labels_ls + labels_pr, loc="upper left", fontsize=8)

        # X轴年份刻度
        if np.issubdtype(df_bar[time_col].dtype, np.datetime64):
            for ax in [ax_eq, ax_ls]:
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.setp(ax_eq.get_xticklabels(), visible=False)

        ax_ls.set_xlabel("Time")

        # 样本外区域高亮
        if df_bar[time_col].max() >= oos_start_ts:
            for ax in [ax_eq, ax_ls]:
                ax.axvspan(oos_start_ts, df_bar[time_col].iloc[-1], color="grey", alpha=0.12)
                ax.axvline(oos_start_ts, color="grey", linestyle="--", linewidth=1)

            ax_eq.text(
                oos_start_ts,
                ax_eq.get_ylim()[1] * 0.9,
                "OOS",
                color="grey",
                fontsize=9,
                ha="left",
                va="top",
            )

        # 底部：年度统计表
        ax_table = fig.add_subplot(gs[2, 0])
        ax_table.axis("off")

        if not yearly_stats.empty:
            cols_show = [
                "Sample", "Year",
                "Return", "LongReturn", "ShortReturn",
                "Sharpe", "MaxDD",
                "NTrades", "AvgHoldBars", "DailyTurnover", "BIPS",
            ]
            table_data = yearly_stats[cols_show].copy()
            for c in [
                "Return", "LongReturn", "ShortReturn", "Sharpe", "MaxDD",
                "AvgHoldBars", "DailyTurnover", "BIPS"
            ]:
                table_data[c] = table_data[c].astype(float).round(3)

            table = ax_table.table(
                cellText=table_data.values,
                colLabels=table_data.columns,
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.4)
            ax_table.set_title("Yearly stats (IS / OOS)", fontsize=11, pad=18)

        fig.suptitle("Vol Trading Backtest (scaled entries)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
        plt.show()

    return trades_df, df_bar, yearly_stats