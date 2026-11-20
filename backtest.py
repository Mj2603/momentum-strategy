"""Rolling and summary analysis of monthly strategy returns."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from momentum_strategy import MONTHS_PER_YEAR, drawdown, performance_metrics


class BacktestAnalyzer:
    """Analysis helpers for the output of :func:`momentum_strategy.run_backtest`."""

    def __init__(self, window: int = 12):
        self.window = window

    def analyze_returns(self, results: pd.DataFrame) -> Dict[str, pd.Series]:
        r = results["net_return"]
        w = self.window
        roll_mean = r.rolling(w).mean() * MONTHS_PER_YEAR
        roll_vol = r.rolling(w).std() * np.sqrt(MONTHS_PER_YEAR)
        return {
            "monthly_returns": r,
            "rolling_ann_mean": roll_mean,
            "rolling_ann_vol": roll_vol,
            "rolling_sharpe": roll_mean / roll_vol,
            "drawdown": drawdown(r),
        }

    def generate_report(self, results: pd.DataFrame) -> Dict[str, float]:
        r = results["net_return"]
        report = performance_metrics(r)
        downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
        report["sortino"] = (
            float(r.mean() / downside * np.sqrt(MONTHS_PER_YEAR)) if downside > 0 else float("nan")
        )
        report["avg_turnover"] = float(results["turnover"].mean())
        report["total_cost"] = float(results["cost"].sum())
        return report

    def plot_analysis(self, results: pd.DataFrame, path: Optional[str] = None):
        import matplotlib.pyplot as plt

        a = self.analyze_returns(results)
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        (1 + a["monthly_returns"]).cumprod().plot(ax=axes[0])
        axes[0].set_ylabel("Growth of 1 (net)")
        a["rolling_sharpe"].plot(ax=axes[1])
        axes[1].set_ylabel(f"Rolling {self.window}m Sharpe")
        (a["drawdown"] * 100).plot(ax=axes[2], color="tab:red")
        axes[2].set_ylabel("Drawdown (%)")
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=120)
        else:
            plt.show()
        return fig
