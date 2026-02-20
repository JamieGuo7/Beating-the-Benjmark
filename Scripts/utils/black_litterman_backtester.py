"""
This script loads LSTM models, and runs inference on them.
This then feeds predictions into Black-Litterman without getting market caps
Then simulates portfolio returns.

This is the same as the main loop, we just skip the training (the long bit!).

We have:
- Data collected
- LSTM Models trained and saved
"""
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import tensorflow as tf

from Scripts.data_pipeline.covariance_calculator import CovarianceCalculator
from Scripts.data_pipeline.features import engineer_features
from Scripts.data_pipeline.data_collector import DataCollector
from Scripts.portfolio_optimisation.portfolio_optimiser import PortfolioOptimiser

class BlackLittermanBacktester:
    def __init__(self, models_dir, preprocessors_dir, forecast_df, tickers, feature_cols,
                 train_date_start, train_date_end, test_date_start, test_date_end,
                 shares_outstanding = None, resolution = 'w', window=30):
        """
        The backtesting happens on date_start, date_end.
        Resolution takes a d, w, or m and sets that as the frequency of backtesting

        Preprocessors have had fit_transform already!
        """
        self.forecast_df = forecast_df

        self.tickers = tickers
        self.feature_cols = feature_cols
        self.resolution = resolution
        self.window = window
        self.train_date_start = train_date_start
        self.train_date_end = train_date_end
        self.test_date_start = test_date_start
        self.test_date_end = test_date_end
        self.data = self.load_data()
        self.train_data = self.data[
            (pd.to_datetime(self.data['Date']) >= train_date_start) &
            (pd.to_datetime(self.data['Date']) <= train_date_end)
            ]
        self.test_data = self.data[
            (pd.to_datetime(self.data['Date']) >= test_date_start) &
            (pd.to_datetime(self.data['Date']) <= test_date_end)
            ]

        if shares_outstanding is not None:
            shares_outstanding['date'] = pd.to_datetime(shares_outstanding['date'])  # explicitly convert
            shares_outstanding = shares_outstanding.set_index(['date', 'ticker'])
            shares_outstanding.index.names = ['date', 'ticker']

            self.shares_outstanding = shares_outstanding
        else:
            self.shares_outstanding = self.fetch_quarterly_shares()

        self.models = self.load_all_models(models_dir)
        self.preprocessors = self.load_all_preprocessors(preprocessors_dir)

    def load_data(self):
        """
        Loads the data, adds relevant features and then returns it
        """
        collector = DataCollector(
            ticker_path=str(project_root / "data" / "ESGU_Tickers.txt"),
            file_path=str(project_root / "data" / "ESGU_LSTM_Ready.csv"),
            period='12y',
            interval='1d'
        )

        data = collector.get_clean_data()
        data = engineer_features(data)
        return data

    def resample_data(self, data):
        # 'Date' needs to be the index for resampling
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')

        # takes first row within resolution period for each ticker
        data_resample = data.groupby('Ticker').resample(self.resolution.upper()).first()

        # Reset index so that 'Date' comes back as a column
        data_resample = data_resample.reset_index()

        # There could be a multiindex error with ticker and date!
        # This could cause duplicate
        return data_resample[['Date', 'Ticker', 'Close'] + self.feature_cols]

    def load_all_models(self, models_dir):
        from Scripts.models.lstm_model import directional_loss
        # Adds each model to a dict, so that we can infer quickly
        models = {}
        for ticker in self.tickers:
            model_path = Path(models_dir) / f"lstm_{ticker}_best.keras"
            if model_path.exists():
                models[ticker] = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'directional_loss': directional_loss}
                )
        print(f"[*] Loaded {len(models)} models")
        return models

    def load_all_preprocessors(self, preprocessors_dir):
        preprocessors = {}
        for ticker in self.tickers:
            prep_path = preprocessors_dir / f"{ticker}_preprocessor.joblib"
            if prep_path.exists():
                preprocessors[ticker] = joblib.load(prep_path)
        print(f"[*] Loaded {len(preprocessors)} preprocessors")
        return preprocessors

    def infer_at_date(self, ticker, date):
        df_test_ticker = self.data[self.data['Ticker'] == ticker]
        df_test_ticker = df_test_ticker[df_test_ticker['Date'] <= date].tail(self.window)

        if len(df_test_ticker) < self.window:
            return None

        X_raw = df_test_ticker[self.feature_cols].values
        X_seq = X_raw.reshape(1, self.window, len(self.feature_cols))

        # We have already fitted from before! preprocessors have had fit_transform
        X_scaled = self.preprocessors[ticker].transform(X_seq)

        pred_scaled = self.models[ticker].predict(X_scaled, verbose=0)
        pred_log = self.preprocessors[ticker].inverse_transform_y(pred_scaled)[0]

        return (np.exp(pred_log) - 1) * 100

    def rolling_covariance(self, date, lookback=252):
        data_to_date = self.data[pd.to_datetime(self.data['Date']) <= date]

        prices = (
            data_to_date[['Date', 'Ticker', 'Close']]
            .drop_duplicates(subset=['Date', 'Ticker'])
            .pivot(index='Date', columns='Ticker', values='Close')
            .sort_index()
            .tail(lookback + 1)
        )

        calc = CovarianceCalculator()
        return calc.from_prices(prices, trading_days=252)

    def get_actual_returns(self, date_start):
        date_end = date_start + pd.offsets.BDay(21)

        window = self.data[
            (pd.to_datetime(self.data['Date']) > date_start) &
            (pd.to_datetime(self.data['Date']) <= date_end)
            ]

        returns = {}
        for ticker, grp in window.groupby('Ticker'):
            grp = grp.sort_values('Date')
            if len(grp) >= 2:
                price_start = grp['Close'].iloc[0]
                price_end = grp['Close'].iloc[-1]

                # Fx Fees
                buy_fee = price_start * 0.0015
                sell_fee = price_end * 0.0015

                net_return = (price_end - price_start - (buy_fee + sell_fee)) / price_start
                returns[ticker] = net_return

        return returns

    def plot_results(self, results_df, backtest_dir, benchmark_df):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        results_df = results_df.copy()
        results_df['date'] = pd.to_datetime(results_df['date'])

        bm_returns = []
        for date in results_df['date']:
            date_end = date + pd.offsets.BDay(21)
            bm_data = benchmark_df[
                (pd.to_datetime(benchmark_df['Date']) > date) &
                (pd.to_datetime(benchmark_df['Date']) <= date_end)
                ].sort_values('Date')

            if len(bm_data) >= 2:
                price_start = bm_data['Close'].iloc[0]
                price_end = bm_data['Close'].iloc[-1]
                buy_fee = price_start * 0.0015
                sell_fee = price_end * 0.0015
                bm_ret = (price_end - price_start - (buy_fee + sell_fee)) / price_start
                bm_returns.append(1 + bm_ret)
            else:
                bm_ret = 0.0

        results_df['benchmark_value'] = bm_returns

        # --- Plot ---
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(13, 8),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )

        ax1.plot(results_df['date'], results_df['portfolio_value'],
                 color='#1f77b4', linewidth=2, label='BL-LSTM Portfolio')
        ax1.plot(results_df['date'], results_df['benchmark_value'],
                 color='#ff7f0e', linewidth=2, linestyle='--', label=f'{benchmark_df['Ticker'].iloc[0]}')
        ax1.axhline(1.0, color='grey', linewidth=0.8, linestyle=':')
        ax1.set_ylabel('Value (£1 base)', fontsize=11)
        ax1.set_title('Monthly Portfolio Return vs ESGU Benchmark', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax1.fill_between(results_df['date'],
                         results_df['portfolio_value'],
                         results_df['benchmark_value'],
                         where=results_df['portfolio_value'] >= results_df['benchmark_value'],
                         alpha=0.15, color='green', label='Outperformance')
        ax1.fill_between(results_df['date'],
                         results_df['portfolio_value'],
                         results_df['benchmark_value'],
                         where=results_df['portfolio_value'] < results_df['benchmark_value'],
                         alpha=0.15, color='red', label='Underperformance')

        # --- Bottom: excess return per month ---
        excess = (results_df['portfolio_value'] - results_df['benchmark_value']) * 100
        ax2.bar(results_df['date'], excess,
                color=['green' if x >= 0 else 'red' for x in excess],
                alpha=0.7, width=20)
        ax2.axhline(0, color='grey', linewidth=0.8)
        ax2.set_ylabel('Excess Return (%)', fontsize=10)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30)
        ax2.grid(True, alpha=0.3)

        # --- Summary stats box ---
        avg_port = (results_df['portfolio_value'] - 1).mean() * 100
        avg_bm = (results_df['benchmark_value'] - 1).mean() * 100
        win_rate = (results_df['portfolio_value'] > results_df['benchmark_value']).mean() * 100
        textstr = (
            f"Avg monthly return: {avg_port:+.2f}%\n"
            f"Avg benchmark:      {avg_bm:+.2f}%\n"
            f"Win rate vs benchmark: {win_rate:.0f}%"
        )
        ax1.text(0.02, 0.97, textstr, transform=ax1.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(backtest_dir / 'backtest_chart.png', dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def fetch_quarterly_shares(self):
        """
        Fetches shares outstanding once per quarter across the test period.
        Note: yfinance only returns current shares — we apply them across quarters
        as an approximation (shares outstanding is stable for large-caps).
        """
        import yfinance as yf

        path = project_root / 'data' / 'shares_outstanding.csv'

        quarterly_dates = pd.date_range(
            start=self.test_date_start,
            end=self.test_date_end,
            freq='QS'
        )

        print(
            f"[*] Fetching shares outstanding for {len(self.tickers)} tickers across {len(quarterly_dates)} quarters...")

        records = []
        for ticker in self.tickers:
            try:
                shares = yf.Ticker(ticker).info.get('sharesOutstanding', None)
                if shares:
                    for date in quarterly_dates:
                        records.append({'date': date, 'ticker': ticker, 'shares': shares})
                else:
                    print(f"[!] No shares data for {ticker}, will fallback to equal weight")
            except Exception as e:
                print(f"[!] Failed {ticker}: {e}")

        df = pd.DataFrame(records)
        df.to_csv(path, index=False)
        print(f"[ok] Shares fetched and saved for {df['ticker'].nunique()} tickers")
        return df.set_index(['date', 'ticker'])

    def get_market_caps(self, tickers, date):
        """
        Market cap = shares outstanding (nearest quarter) x close price at date.
        Falls back to 1e9 if data is missing.
        """
        available_quarters = self.shares_outstanding.index.get_level_values('date').unique()
        nearest_quarter = available_quarters[available_quarters <= date].max()

        mcaps = {}
        for ticker in tickers:
            try:
                shares = self.shares_outstanding.loc[(nearest_quarter, ticker), 'shares']
                close = self.data[
                    (self.data['Ticker'] == ticker) &
                    (pd.to_datetime(self.data['Date']) <= date)
                    ]['Close'].iloc[-1]
                mcaps[ticker] = shares * close
            except (KeyError, IndexError):
                mcaps[ticker] = 1e9

        return pd.Series(mcaps)

    def run(self, rebalance_dates):
        """
        we backtest on rebalance dates that are out of sample.
        This means that we can use test_rmse from our relevant forecast_df.

        Tradeoffs (for time):
        - We are using the same rmse (to avoid retraining) throughout the out of sample period
        - Using Market Caps = Shares outstanding x Close Price instead of getting them
        - Getting shares outstanding every quarter of the out of sample period
        - Thus assuming that shares outstanding doesn't change much
        """

        results = []
        weights_history = {}
        current_weights = None

        rmses = dict(zip(self.forecast_df['ticker'], self.forecast_df['test_rmse']))

        for i, date in enumerate(rebalance_dates):
            print(f"Backtesting on {date.date()}")
            next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None

            forecasts_at_date = {}
            for ticker in self.tickers:
                pred = self.infer_at_date(ticker, date)
                if pred is not None:
                    forecasts_at_date[ticker] = pred

            # Only keep those for which we have an rmse for.
            forecasts_at_date = {t: v for t, v in forecasts_at_date.items() if t in rmses}

            results_df = pd.DataFrame({
                'ticker': list(forecasts_at_date.keys()),
                'forecast_return': list(forecasts_at_date.values()),
                'test_rmse': [rmses[t] for t in forecasts_at_date],
                'target_date': date + pd.offsets.BDay(21)
            })

            print(f"This is the results: {results_df}")

            cov_matrix = self.rolling_covariance(date, lookback=252)
            mcaps = self.get_market_caps(list(forecasts_at_date), date)

            print(f"This is the covariance matrix: {cov_matrix}")
            print(f"This is the market caps: {mcaps}")

            # keys of forecasts_at_date are the tickers
            optimiser = PortfolioOptimiser(cov_matrix)
            result = optimiser.optimise_black_litterman(
                list(forecasts_at_date.keys()), results_df,
                mcaps = mcaps,
                verbose = False
            )
            new_weights = result['weights']

            if current_weights is not None and next_date is not None:
                actual_returns = self.get_actual_returns(date)
                port_return = sum(
                    current_weights.get(t, 0) * actual_returns.get(t, 0)
                    for t in actual_returns
                )

                results.append({
                    'date': date,
                    'port_return': port_return,
                    'portfolio_value': 1 + port_return,
                    'n_assets': sum(1 for w in new_weights.values() if w > 0.001)
                })

            current_weights = new_weights
            weights_history[date] = new_weights

        return pd.DataFrame(results), weights_history



project_root = Path(__file__).resolve().parents[2]

models_dir       = project_root / 'models'
preprocessors_dir = project_root / 'models' / 'preprocessors'
forecast_df      = pd.read_csv(project_root / 'results' / 'results.csv')
tickers          = sorted(pd.read_csv(project_root / 'data' / 'ESGU_LSTM_Ready.csv')['Ticker'].unique().tolist())
feature_cols = ['dist_sma200', 'ret_21d', 'momentum_quality', 'dist_high52w', 'efficiency_ratio', 'adx_slope', 'vol_ratio','NATR']

train_date_start = '2014-02-18'
train_date_end   = '2023-09-08'
test_date_start  = '2025-01-01'
test_date_end    = '2026-01-20'

shares_path = project_root / 'data' / 'shares_outstanding.csv'
shares = pd.read_csv(shares_path)

rebalance_dates = pd.date_range(
    start=test_date_start,
    end=test_date_end,
    freq='W-MON'  # each monday
).tolist()

backtester = BlackLittermanBacktester(
    models_dir=models_dir,
    preprocessors_dir=preprocessors_dir,
    forecast_df=forecast_df,
    tickers=tickers,
    feature_cols=feature_cols,
    train_date_start=train_date_start,
    train_date_end=train_date_end,
    test_date_start=test_date_start,
    test_date_end=test_date_end,
    shares_outstanding = shares,
    resolution='w',
    window=30
)

results_df, weights_history = backtester.run(rebalance_dates)

backtest_dir = project_root / 'results' / f'backtest_{test_date_start}_to_{test_date_end}'

weights_df = pd.DataFrame(weights_history).T
weights_df.to_csv(backtest_dir / 'weights_history.csv')
results_df.to_csv(backtest_dir / 'backtest_results.csv', index=False)

results_df = pd.read_csv(backtest_dir / 'backtest_results.csv')
weights_df = pd.read_csv(backtest_dir / 'weights_history.csv')

benchmark_df = pd.read_csv(project_root / 'data' / 'ESGU_benchmark.csv')

backtester.plot_results(results_df, backtest_dir, benchmark_df)

