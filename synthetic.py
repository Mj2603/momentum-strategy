"""Seeded SYNTHETIC monthly price panels with a known ground truth.

Monthly log return of stock i in month t:

    log_r[t, i] = mu_i + sigma * eps[t, i],   eps ~ iid N(0, 1)

* ``drift_dispersion = 0``: every stock has the same mu, so past returns carry
  no information about future returns and the expected momentum return is 0.
* ``drift_dispersion > 0``: mu_i ~ N(base_mu, drift_dispersion^2) is fixed per
  stock, so past winners tend to keep winning and momentum earns a positive
  expected return by construction.

These are not market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_monthly_prices(
    n_stocks: int = 100,
    n_months: int = 240,
    sigma: float = 0.08,
    base_mu: float = 0.005,
    drift_dispersion: float = 0.0,
    seed: int = 0,
    start: str = "2000-01-31",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mu = base_mu + drift_dispersion * rng.standard_normal(n_stocks)
    log_r = mu + sigma * rng.standard_normal((n_months, n_stocks))
    log_p = np.vstack([np.zeros(n_stocks), np.cumsum(log_r, axis=0)])
    index = pd.date_range(start=start, periods=n_months + 1, freq="ME")
    cols = [f"SYN{i:03d}" for i in range(n_stocks)]
    return pd.DataFrame(100.0 * np.exp(log_p), index=index, columns=cols)
