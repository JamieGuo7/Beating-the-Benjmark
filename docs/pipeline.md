# Pipeline

Five stages, each one its own folder under [`trader/`](../trader/). This
page walks them in order, says what each one is responsible for, and where each
stage can go wrong.

```
data/raw/index/esgu_tickers.txt
        │
        │  1. ACQUISITION          trader/data/collector.py
        ▼
data/raw/prices/esgu_ohlcv_daily.csv        802,455 rows, 276 tickers, 12 years
        │
        │  2. FEATURES             trader/data/features.py
        ▼
8 technical features + the Forward Log Return target
        │
        │  3. TRAINING             trader/training/{trainer,batch}.py
        ▼
runs/{date_range}_{version_name}_{model}/           274 models + 274 preprocessors + results.csv
        │
        │  4. PORTFOLIO            trader/portfolio/{covariance,optimiser}.py
        ▼
weights per Rebalance Date        Black-Litterman and Efficient Frontier
        │
        │  5. BACKTEST             trader/backtesting
        ▼
runs/{run}/backtests/{period}/    weights_history.csv, significance
```

## Why it is split this way

The idea of the boundaries is set to be a clear checkpoint so that when things are broken, you don't have to redo a stage from before. 

Between stages, we should verify our output and ensure that it is as expected before feeding it to the next stage. 
---

## 1. Acquisition

[`trader/data/collector.py`](../trader/data/collector.py) - `DataCollector`

Reads the ticker list, downloads 12 years of daily bars from Yahoo Finance in one
`yf.download` call with `auto_adjust=True`, then tidies the result from Yahoo's
multi-index wide format into long format: one row per (Date, Ticker).

Long format is the right shape here because tickers have different histories -
`FLUT` has 570 rows, most have 3,019 so therefore a wide frame would be mostly padding.

**Limitations**
- Yahoo can change the past. So we need to save a copy of the data we used, otherwise running the same backtest again later could give a different result.
- The module reads the whole data file at module level.

## 2. Feature engineering

[`trader/data/features.py`](../trader/data/features.py) - `engineer_features(df)`

Eight features and one target, computed per ticker from OHLCV. Every feature is a
ratio, difference or normalisation - never a price level - for the reasons in
[`data.md`](data.md#what-the-data-looks-like). Full list and rationale in
[`modelling.md`](modelling.md#the-features).

The target is the **Forward Log Return** over 21 trading days:

```python
df[TARGET_COL] = np.log(df['Close'].shift(-FORECAST_HORIZON_DAYS) / df['Close'])
```

21 trading days is about one calendar month. The horizon lives in
[`trader/config.py`](../trader/config.py) as a single constant.

**Limitations**
- `dist_sma200` needs 200 rows and `dist_high52w` needs 252 before they produce
  anything, so the first year of every ticker is dropped by the subsequent
  `dropna`. This reduces the training data we can have.

## 3. Training

[`trader/training/trainer.py`](../trader/training/trainer.py) - `TickerModelTrainer`
[`trader/training/batch.py`](../trader/training/batch.py) - `BatchTrainer`

Per ticker: engineer features, drop rows with NaNs, build sequence windows of
30 days for the LSTM, split 80/10/10, fit the preprocessor on train only,
train an LSTM, evaluate on test, save the model and the fitted preprocessor,
and produce one forward Forecast.

`BatchTrainer` loops that over all tickers, skips failures, and aggregates into
`results.csv`. It asks for `yes/no` confirmation first, because a full run is
274 models and around 2 hours.

Two decisions here:

**The split is chronological and never shuffled.** This is a time series. A
shuffled split lets the model see the future and so we have leakage.

**Train and validation sets are separated by a 21-sample gap at their boundary.** This was introduced after we identified a data leakage bug. Since the label for day *t* is calculated using the price on day *t+21*, the final 21 training labels would otherwise depend on prices that appear as input features in the validation set. Removing these 21 samples prevents information from the validation period from leaking into the training data.

```python
X_train = X[:train_size - self.horizon]
X_val   = X[train_size:train_size + val_size - self.horizon]
```

**Limitations**
- The 30-day sequences are built before splitting the data, so sequences near
  the split boundary mix training and test rows. This leak is left in on
  purpose since fixing it now would confound results with the retrain, and we'd
  no longer know which change caused what. This will be patched.
- A saved preprocessor remembers *how many* features it expects, but not
  *which ones* or their order. Swap the feature list and reload an old model,
  and it'll accept the wrong data. This is why we do version tracking is essential.

## 4. Portfolio construction

[`trader/portfolio/covariance.py`](../trader/portfolio/covariance.py) - `CovarianceCalculator`
[`trader/portfolio/optimiser.py`](../trader/portfolio/optimiser.py) - `PortfolioOptimiser`

Both steps use [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/).

1. **Market-implied prior.** Reverse-engineer the market's implied return
   forecasts from the market-cap-weighted portfolio, at risk aversion δ = 2.5.
   Market cap = `shares_outstanding × close`, defaulting to $1e9 when unknown.
2. **Views.** Each ticker's Forecast becomes an absolute Black-Litterman view.
3. **Uncertainty (Ω).** Ω is diagonal, built from each model's own test RMSE
   scaled by that ticker's variance and τ = 0.025. Models that fit their
   ticker poorly get wide, low-confidence views and models that fit well get
   tight ones. This beats treating all views as equally confident, which
   would ignore how good each model actually is.
4. **Posterior → weights.** Blend prior and views, then run
   [Efficient Frontier](https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html)
   `max_sharpe` with a no-short constraint.

`generate_diagnostics` / `print_report` exist because Black-Litterman can fail when
Ω is far larger than τΣ so that the views are ignored and you get the market
portfolio back with extra steps. The uncertainty ratio in the report is there to
catch that.

**Limitations**
- Ω is the **frozen test RMSE** from training, not a rolling realised error. The
  confidence assigned in 2025 is derived from how the model did on its 2024 test
  split.
- No-short plus `max_sharpe` on 274 assets tends to produce concentrated portfolios.

## Running it

```bash
python -m trader.main                       # train all tickers (slow, prompts first)
python -m trader.data.collector             # refresh prices
```

Run from the repository root so `trader.*` resolves. Dates are hardcoded in
`main.py` rather than passed in.