"""Jegadeesh-Titman style 12-1 cross-sectional momentum on monthly prices.

Conventions (all on month-end prices P):

* Signal at month-end t: ``P[t - skip] / P[t - lookback] - 1``. With the
  defaults (lookback=12, skip=1) this is the return from month-end t-12 to
  month-end t-1, so the most recent month is excluded.
* Portfolio formed at month-end t is held over month t+1 and earns
  ``sum_i w[t, i] * r[t+1, i]`` where ``r[t+1] = P[t+1] / P[t] - 1``.
  Weights at t only use prices up to and including t.
* Long leg: equal weights summing to +1 over the ``top_n`` highest signals.
  Short leg: equal weights summing to -1 over the ``top_n`` lowest signals.
  Gross exposure is 2, net exposure is 0.
* Transaction cost: ``cost_bps / 1e4 * sum_i |w[t, i] - w[t-1, i]|``, charged
  in the month the trade happens (including the initial entry). Turnover is
  measured against the previous target weights, ignoring intra-month drift.
* A held stock whose next-month return is missing (e.g. delisted) earns 0
  for that month. Delisting returns are not modelled.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    """Last available price in each calendar month (index = month-end)."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex")
    return prices.resample("ME").last()


def momentum_signal(monthly_prices: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """12-1 momentum: P[t-skip] / P[t-lookback] - 1 for every month t.

    Rows without enough history are NaN.
    """
    if not 0 <= skip < lookback:
        raise ValueError("need 0 <= skip < lookback")
    return monthly_prices.shift(skip) / monthly_prices.shift(lookback) - 1.0


def form_weights(signal_row: pd.Series, top_n: int) -> pd.Series:
    """Equal-weight long top_n / short bottom_n weights from one signal row.

    Stocks with a NaN signal get weight 0. Returns all zeros if fewer than
    ``2 * top_n`` stocks have a valid signal. Ties are broken by column order.
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    weights = pd.Series(0.0, index=signal_row.index)
    valid = signal_row.dropna()
    if len(valid) < 2 * top_n:
        return weights
    ranked = valid.sort_values(ascending=False, kind="mergesort")
    weights[ranked.index[:top_n]] = 1.0 / top_n
    weights[ranked.index[-top_n:]] = -1.0 / top_n
    return weights


def run_backtest(
    monthly_prices: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
    top_n: int = 10,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Monthly long-short momentum backtest.

    Returns a DataFrame indexed by the month in which the return is earned,
    with columns: gross_return, turnover, cost, net_return, n_long, n_short.
    """
    signal = momentum_signal(monthly_prices, lookback, skip)
    fwd_returns = (monthly_prices / monthly_prices.shift(1) - 1.0).shift(-1)

    rows = []
    prev_w = pd.Series(0.0, index=monthly_prices.columns)
    dates = monthly_prices.index
    for k in range(len(dates) - 1):
        t = dates[k]
        w = form_weights(signal.loc[t], top_n)
        if (w != 0).sum() == 0:
            continue
        r_next = fwd_returns.loc[t].fillna(0.0)
        gross = float((w * r_next).sum())
        turnover = float((w - prev_w).abs().sum())
        cost = cost_bps / 1e4 * turnover
        rows.append(
            {
                "date": dates[k + 1],
                "gross_return": gross,
                "turnover": turnover,
                "cost": cost,
                "net_return": gross - cost,
                "n_long": int((w > 0).sum()),
                "n_short": int((w < 0).sum()),
            }
        )
        prev_w = w
    if not rows:
        raise ValueError("not enough history to form any portfolio")
    return pd.DataFrame(rows).set_index("date")


def drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown of the compounded equity curve, starting from 1 before the first return."""
    equity = (1.0 + returns).cumprod()
    peak = np.maximum(equity.cummax(), 1.0)
    return equity / peak - 1.0


def performance_metrics(returns: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> Dict[str, float]:
    """Summary statistics for a series of periodic simple returns (rf = 0)."""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        raise ValueError("empty return series")
    growth = float((1.0 + r).prod())
    vol = float(r.std(ddof=1)) if n > 1 else 0.0
    mean = float(r.mean())
    return {
        "months": float(n),
        "total_return": growth - 1.0,
        "cagr": growth ** (periods_per_year / n) - 1.0,
        "ann_mean": mean * periods_per_year,
        "ann_vol": vol * np.sqrt(periods_per_year),
        "sharpe": (mean / vol * np.sqrt(periods_per_year)) if vol > 0 else float("nan"),
        "t_stat_mean": (mean / (vol / np.sqrt(n))) if vol > 0 else float("nan"),
        "max_drawdown": float(drawdown(r).min()),
        "hit_rate": float((r > 0).mean()),
    }


def fetch_prices(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily adjusted closes with yfinance (optional dependency, needs network)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install the 'data' extra: pip install -e '.[data]'") from exc
    data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise RuntimeError("no price data returned (network blocked or bad symbols)")
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    return close.dropna(how="all")


class MomentumStrategy:
    """Convenience wrapper around :func:`run_backtest`."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        lookback: int = 12,
        skip: int = 1,
        top_n: int = 10,
        cost_bps: float = 10.0,
    ):
        self.symbols = symbols
        self.lookback = lookback
        self.skip = skip
        self.top_n = top_n
        self.cost_bps = cost_bps

    def backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        prices: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Run the backtest on ``prices`` (daily or monthly), or download them."""
        if prices is None:
            if not self.symbols or start_date is None or end_date is None:
                raise ValueError("pass prices, or symbols plus start_date and end_date")
            prices = fetch_prices(self.symbols, start_date, end_date)
        monthly = to_month_end(prices)
        return run_backtest(monthly, self.lookback, self.skip, self.top_n, self.cost_bps)

    def calculate_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        out = performance_metrics(results["net_return"])
        out["avg_turnover"] = float(results["turnover"].mean())
        out["total_cost"] = float(results["cost"].sum())
        return out

    def plot_results(self, results: pd.DataFrame, path: Optional[str] = None):
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        (1 + results["net_return"]).cumprod().plot(ax=ax1, label="net")
        (1 + results["gross_return"]).cumprod().plot(ax=ax1, label="gross")
        ax1.set_ylabel("Growth of 1")
        ax1.legend()
        (drawdown(results["net_return"]) * 100).plot(ax=ax2, color="tab:red")
        ax2.set_ylabel("Drawdown (%)")
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=120)
        else:
            plt.show()
        return fig
