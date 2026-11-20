import numpy as np
import pandas as pd
import pytest

from backtest import BacktestAnalyzer
from momentum_strategy import (
    MomentumStrategy,
    drawdown,
    form_weights,
    momentum_signal,
    performance_metrics,
    run_backtest,
    to_month_end,
)
from synthetic import synthetic_monthly_prices


def _panel(values, start="2010-01-31"):
    arr = np.asarray(values, dtype=float)
    idx = pd.date_range(start, periods=arr.shape[0], freq="ME")
    return pd.DataFrame(arr, index=idx, columns=[f"S{i}" for i in range(arr.shape[1])])


def test_signal_matches_closed_form_and_skips_last_month():
    prices = synthetic_monthly_prices(n_stocks=5, n_months=30, seed=1)
    sig = momentum_signal(prices, lookback=12, skip=1)
    t = 20
    expected = prices.iloc[t - 1] / prices.iloc[t - 12] - 1
    pd.testing.assert_series_equal(sig.iloc[t], expected, check_names=False)
    assert sig.iloc[:12].isna().all().all()


def test_signal_ignores_current_month_and_future_prices():
    prices = synthetic_monthly_prices(n_stocks=6, n_months=30, seed=2)
    t = 18
    base = momentum_signal(prices).iloc[t]
    shocked = prices.copy()
    shocked.iloc[t:] *= np.linspace(0.1, 10, prices.shape[1])  # change month t and later
    pd.testing.assert_series_equal(momentum_signal(shocked).iloc[t], base)


def test_weights_long_short_equal_weight():
    sig = pd.Series([0.5, -0.2, 0.1, 0.9, -0.7, np.nan], index=list("abcdef"))
    w = form_weights(sig, top_n=2)
    assert w["d"] == w["a"] == 0.5
    assert w["e"] == w["b"] == -0.5
    assert w["c"] == 0 and w["f"] == 0
    assert w.sum() == pytest.approx(0.0)
    assert w.abs().sum() == pytest.approx(2.0)
    assert (form_weights(sig, top_n=3) == 0).all()  # only 5 valid signals < 6


def test_backtest_return_and_costs_by_hand():
    # 2 stocks, lookback 2, skip 1: signal at t = P[t-1]/P[t-2]-1.
    prices = _panel([[100, 100], [110, 90], [120, 100], [108, 110], [108, 121]])
    res = run_backtest(prices, lookback=2, skip=1, top_n=1, cost_bps=50)
    # t=2: signal = (0.10, -0.10) -> long S0, short S1; month-3 returns: -10%, +10%
    # t=3: signal = (120/110-1, 100/90-1) = (0.0909, 0.111) -> long S1, short S0;
    #      month-4 returns: 0%, +10%
    assert list(res.index) == list(prices.index[3:5])
    assert res["gross_return"].iloc[0] == pytest.approx(-0.10 - 0.10)
    assert res["gross_return"].iloc[1] == pytest.approx(0.10 - 0.0)
    assert res["turnover"].tolist() == pytest.approx([2.0, 4.0])
    assert res["cost"].tolist() == pytest.approx([0.01, 0.02])
    np.testing.assert_allclose(res["net_return"], res["gross_return"] - res["cost"])


def test_cost_reduces_return_by_exactly_cost_times_turnover():
    prices = synthetic_monthly_prices(n_stocks=40, n_months=60, seed=3)
    a = run_backtest(prices, top_n=5, cost_bps=0)
    b = run_backtest(prices, top_n=5, cost_bps=25)
    np.testing.assert_allclose(a["net_return"] - b["net_return"], 25e-4 * a["turnover"])


def test_missing_next_return_is_zero_not_dropped():
    prices = _panel([[100, 100, 100], [110, 100, 90], [110, 100, 90], [np.nan, 100, 81]])
    res = run_backtest(prices, lookback=2, skip=1, top_n=1, cost_bps=0)
    # long S0 (missing next price -> 0), short S2 (-10%) -> gross +10%
    assert res["gross_return"].iloc[0] == pytest.approx(0.10)


def test_metrics_closed_form_constant_return():
    r = pd.Series([0.01] * 24)
    m = performance_metrics(r)
    assert m["cagr"] == pytest.approx(1.01 ** 12 - 1)
    assert m["total_return"] == pytest.approx(1.01 ** 24 - 1)
    assert m["ann_vol"] == pytest.approx(0.0, abs=1e-15)
    assert m["max_drawdown"] == 0.0
    assert m["hit_rate"] == 1.0


def test_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20])
    dd = drawdown(r)
    assert dd.tolist() == pytest.approx([0.0, -0.5, -0.4])
    assert drawdown(pd.Series([-0.2])).iloc[0] == pytest.approx(-0.2)


def test_sharpe_annualises_monthly():
    r = pd.Series([0.02, 0.0] * 30)
    m = performance_metrics(r)
    assert m["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(12))


def test_no_momentum_under_null_and_positive_under_persistent_drift():
    null = [
        run_backtest(synthetic_monthly_prices(seed=s), top_n=10, cost_bps=0)["gross_return"].mean()
        for s in range(10)
    ]
    # average over 10 seeds should be close to zero (expected value is exactly zero)
    assert abs(np.mean(null)) < 0.003
    drift = run_backtest(
        synthetic_monthly_prices(drift_dispersion=0.01, seed=0), top_n=10, cost_bps=0
    )["gross_return"]
    assert performance_metrics(drift)["t_stat_mean"] > 3


def test_wrapper_and_analyzer_on_daily_prices():
    monthly = synthetic_monthly_prices(n_stocks=30, n_months=36, seed=4)
    daily = monthly.resample("D").ffill()
    pd.testing.assert_frame_equal(to_month_end(daily), monthly, check_freq=False)
    strat = MomentumStrategy(top_n=5)
    res = strat.backtest(prices=daily)
    m = strat.calculate_metrics(res)
    assert m["months"] == len(res)
    rep = BacktestAnalyzer().generate_report(res)
    assert rep["sharpe"] == pytest.approx(m["sharpe"])
