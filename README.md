# Momentum Strategy (12-1)

A small Python implementation of the Jegadeesh-Titman (1993) cross-sectional momentum strategy: each month, rank stocks by their return from month t-12 to month t-1 (skipping the most recent month), go long the top N and short the bottom N with equal weights, hold for one month, and charge a transaction cost proportional to turnover. The code is tested against hand-computed cases and against simulated data where the correct answer is known. The results below come from seeded synthetic data, not market data, because this repository is meant to be reproducible offline.

## What is implemented

- `momentum_strategy.py`
  - `momentum_signal`: `P[t-1] / P[t-12] - 1` on month-end prices (lookback and skip configurable).
  - `form_weights`: equal-weight long top N (+1 total) / short bottom N (-1 total); stocks without a valid signal are excluded.
  - `run_backtest`: weights formed at month-end t use only prices up to t and earn the month t+1 return. Cost = `cost_bps / 10000 * sum |w_t - w_{t-1}|`, including the initial entry.
  - `performance_metrics`: annualised mean and volatility (x12, x sqrt 12), Sharpe (rf = 0), t-statistic of the mean, CAGR, max drawdown, hit rate.
  - `MomentumStrategy`: wrapper that accepts daily or monthly prices, or downloads prices with yfinance (optional, needs network).
- `backtest.py`: `BacktestAnalyzer` with rolling 12-month mean, volatility and Sharpe, drawdown series, Sortino ratio and plots.
- `synthetic.py`: seeded SYNTHETIC price panels with constant per-stock drift plus i.i.d. noise.
- `run_experiment.py`: runs the scenarios below and writes `results/synthetic_results.md` and `results/synthetic_results.csv`.
- `tests/`: 11 tests (closed-form signal, no use of the skip month or future prices, weight constraints, a backtest worked by hand, exact cost accounting, metric formulas, zero expected return under the null, positive return when momentum is built in).

## How to run

```bash
git clone https://github.com/Mj2603/momentum-strategy.git
cd momentum-strategy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python run_experiment.py
```

To use real prices (needs network access and the optional dependency):

```bash
pip install -e ".[data]"
```

```python
from momentum_strategy import MomentumStrategy

strategy = MomentumStrategy(["AAPL", "MSFT", "JPM", "XOM"], top_n=1, cost_bps=10)
results = strategy.backtest("2010-01-01", "2023-12-31")   # downloads with yfinance
print(strategy.calculate_metrics(results))
```

Or pass your own price table: `strategy.backtest(prices=df)` where `df` has a `DatetimeIndex` and one column per stock.

## Results (SYNTHETIC data)

Data source: SYNTHETIC, `synthetic.py`, seed 42, 100 stocks, 240 months, 8% monthly volatility. Not market data. Long top 10 / short bottom 10, monthly rebalance. `null` has no drift differences between stocks (true expected return 0); `drift` gives each stock a fixed drift with 1%/month dispersion, so momentum is profitable by construction. Turnover is the sum of absolute weight changes per month (gross exposure is 2).

| scenario | cost_bps | months | ann_mean | ann_vol | sharpe | t_stat | max_drawdown | avg_turnover |
|---|---|---|---|---|---|---|---|---|
| null | 0 | 228 | -2.48% | 12.28% | -0.20 | -0.88 | -53.79% | 1.26 |
| null | 10 | 228 | -4.00% | 12.28% | -0.33 | -1.42 | -63.04% | 1.26 |
| null | 50 | 228 | -10.05% | 12.30% | -0.82 | -3.56 | -88.02% | 1.26 |
| drift | 0 | 228 | 8.81% | 11.83% | 0.75 | 3.25 | -22.62% | 1.18 |
| drift | 10 | 228 | 7.39% | 11.83% | 0.62 | 2.72 | -25.79% | 1.18 |
| drift | 50 | 228 | 1.71% | 11.84% | 0.14 | 0.63 | -43.79% | 1.18 |

Reading: under the null the gross result is not significantly different from zero (t = -0.88), and costs alone make it negative. With built-in momentum the strategy is detected (t = 3.25 gross), and about 1.2 units of monthly turnover cost about 1.4 to 1.5 percentage points a year per 10 bps.

## Limitations

- No results on real market data are reported here.
- Survivorship bias: a hand-picked list of current large-cap tickers (as in the yfinance example) excludes delisted firms and overstates historical performance. A proper test needs a point-in-time universe such as CRSP.
- Delisting returns are not modelled; a held stock with a missing next-month price earns 0 for that month.
- One-month holding period only (no overlapping J/K portfolios as in the original paper), equal weights, no value weighting, no size or price filters.
- Turnover ignores intra-month weight drift; costs are a flat rate with no market impact, borrow fees or short-sale constraints.
- The synthetic model (constant drift, i.i.d. normal noise) has none of the crash risk or time variation of real momentum.

## Reference

Jegadeesh, N. and Titman, S. (1993). Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. *Journal of Finance*, 48(1), 65-91.

## Author

Mrityunjay Balkrishnan - [GitHub](https://github.com/Mj2603)

For educational use. Not investment advice.
