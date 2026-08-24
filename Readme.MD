# Algorithmic Trading Machine

An end-to-end quantitative trading pipeline that combines technical analysis, factor models, and unsupervised machine learning to build and backtest a monthly-rebalanced momentum portfolio, benchmarked against a simple SPY buy-and-hold strategy.

## Project Overview

This project explores whether a momentum-based stock selection strategy — built using technical indicators, Fama-French risk factors, and K-Means clustering — can outperform a passive S&P 500 benchmark. The pipeline downloads historical price data for S&P 500 constituents, engineers a rich feature set, clusters stocks by momentum characteristics each month, and constructs an optimized portfolio using Modern Portfolio Theory.

## Pipeline Architecture

The project is organized into six sequential stages, each in its own notebook, with intermediate results persisted as pickle files to keep the workflow modular and reproducible:

| Stage | Notebook | Description |
|---|---|---|
| 1 | `01_data_download.ipynb` | Scrapes the current S&P 500 ticker list from Wikipedia and downloads 8 years of daily OHLCV data via `yfinance`. |
| 2 | `02_creating_indicators.ipynb` / `03_indicator_engineering.ipynb` | Engineers technical indicators per stock: Garman-Klass volatility, RSI, Bollinger Bands, ATR, MACD, and dollar volume. |
| 3 | `03_indicator_engineering.ipynb` | Resamples daily data to monthly, applies a rolling liquidity filter to keep the top ~150 most liquid stocks each month, and calculates multi-horizon returns (1, 2, 3, 6, 9, 12 months). |
| 4 | `04_fama_french.ipynb` | Downloads the Fama-French 5-factor dataset and estimates each stock's rolling factor exposures (Market, Size, Value, Profitability, Investment) using `RollingOLS`, with a 1-month lag to avoid lookahead bias. |
| 5 | `05_kMeansClustering.ipynb` | Applies K-Means clustering (with fixed RSI-based centroid initialization) to group stocks into 4 clusters each month based on their full feature set. |
| 6 | `06_Portfolio_Optimization.ipynb` | Selects the highest-RSI cluster each month, computes optimal portfolio weights via Efficient Frontier / Max Sharpe Ratio optimization (`PyPortfolioOpt`), and backtests the resulting monthly-rebalanced portfolio against SPY. |

**Key engineering details:**
- All signals are shifted forward in time (via `.shift()` / `DateOffset`) before being used for portfolio decisions, to prevent lookahead bias.
- A minimum-observation threshold filters out stocks with insufficient history before ranking or regression.
- Portfolio optimization falls back to equal-weighting if Max Sharpe optimization fails for a given month (e.g., due to numerical instability in the covariance matrix).

## Key Results

The strategy was backtested from 2020 to 2026 and compared against a SPY buy-and-hold benchmark:

| Metric | Momentum Strategy | SPY Buy & Hold |
|---|---|---|
| CAGR | 6.96% | 15.90% |
| Annualized Volatility | 19.37% | 16.87% |
| Sharpe Ratio | 0.35 | 0.87 |
| Max Drawdown | -34.35% | -24.50% |

**The strategy underperformed the benchmark across every metric** — lower returns, higher volatility, and a deeper drawdown.

## Key Learnings

This result is itself a useful finding, not a failure of the pipeline. A few takeaways:

- **Selecting the highest-RSI cluster likely selects overbought stocks.** RSI above 70 is traditionally interpreted as a mean-reversion signal (a stock is "due for a pullback") rather than a guarantee of continued momentum. This may explain the strategy's underperformance relative to simply holding the index.
- **Lookahead bias is easy to introduce accidentally.** A significant part of this project involved correctly time-shifting rolling betas and cluster-based signals so that decisions are only made using information that would have genuinely been available at that point in time.
- **Backtesting doesn't account for transaction costs or slippage.** Real-world monthly rebalancing would incur trading costs that are not reflected in these results, meaning actual performance would likely be worse than shown here.
- **Rigorous backtesting is essential before trusting a strategy hypothesis.** This project reinforced that a strategy which sounds intuitively reasonable (buy high-momentum stocks) still needs to be validated empirically against a simple benchmark before drawing conclusions.

## Tech Stack

- **Data & Analysis:** `pandas`, `numpy`, `yfinance`, `pandas_datareader`
- **Technical Indicators:** `pandas_ta`
- **Statistics:** `statsmodels` (RollingOLS)
- **Machine Learning:** `scikit-learn` (KMeans)
- **Portfolio Optimization:** `PyPortfolioOpt`
- **Visualization:** `matplotlib`

## How to Run

1. Clone the repository and create a virtual environment:
   ```bash
   conda create -n trade-env python=3.12
   conda activate trade-env
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks in order (01 through 06). Each notebook loads its input from the `data/` folder (produced by the previous stage) and saves its output there for the next stage.

## Possible Extensions

- Compare the low-RSI cluster (potential mean-reversion candidates) against the high-RSI cluster used here.
- Incorporate transaction costs into the backtest for a more realistic performance estimate.
- Extend the historical window and test sensitivity to the number of clusters and rolling window lengths.