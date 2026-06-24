# Cointegrated Pairs Trading Engine: A Statistical Arbitrage Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end, object-oriented algorithmic trading platform implemented in Python. It evaluates, captures, and backtests market-neutral statistical arbitrage opportunities using economically linked equities.

  ## Features
* **Automated Data Infrastructure:** Real-time data pipeline mapping with zero-authentication dependencies via `yfinance`.
* **Engle-Granger Two-Step Routine:** Full Ordinary Least Squares (OLS) calculation tracking residuals alongside Augmented Dickey-Fuller (ADF) stationarity validation.
* **Ornstein-Uhlenbeck Modeling:** Continuous estimation of the mean-reversion half-life coefficient to optimize position holding boundaries.
* **Friction-Adjusted Backtests:** Advanced simulation engine including transactional slip assumptions (basis points tracking).

## Statistical Specifications

### Spread Construction
The pricing residual equilibrium is modeled as:
$$Spread_t = Y_t - (\beta \times X_t) - \alpha$$

### Z-Score Normalization
$$Z_t = \frac{Spread_t - \mu_{\text{Spread}}}{\sigma_{\text{Spread}}}$$

## How To Run Locally
```bash
git clone [https://github.com/YourUsername/pairs-trading-statistical-arbitrage.git](https://github.com/YourUsername/pairs-trading-statistical-arbitrage.git)
cd pairs-trading-statistical-arbitrage
pip install -r requirements.txt
python -c "import src.engine as e; print('System Bootstrapped')"
