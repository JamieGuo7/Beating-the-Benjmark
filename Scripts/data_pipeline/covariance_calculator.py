import pandas as pd
import numpy as np
from pypfopt import risk_models

class CovarianceCalculator:
    """ Calculate covariance matrices from existing price/return data. """

    def __init__(self):
        self.cov_matrix = None

    def from_prices(self, prices_df, trading_days=21):
        """ Calculate monthly covariance from price data. """
        print(f"[*] Calculating covariance from prices...")

        # Calculate daily returns
        daily_returns = prices_df.pct_change(fill_method=None)
        daily_returns = daily_returns.dropna(how='all')
        daily_returns = daily_returns.dropna(axis=1, how='any')

        if daily_returns.shape[1] < 2:
            raise ValueError("Need at least 2 valid tickers")

        print(f"[ok] Valid tickers: {daily_returns.shape[1]}")

        # Calculate covariance
        daily_cov = daily_returns.cov()
        scaled_cov = daily_cov * trading_days

        self.cov_matrix = scaled_cov

        print(f"[ok] Covariance matrix: {scaled_cov.shape}")
        return scaled_cov

    def from_long_data(self, data, price_col='Close', ticker_col='Ticker',
                       date_col='Date', trading_days=21):
        """ Calculate covariance from long-format ticker data. """
        print(f"[*] Calculating covariance from long-format data...")

        # Handle date column
        if date_col in data.columns:
            data = data.copy()
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.set_index(date_col)

        # Pivot to wide format
        print("🔄 Pivoting to wide format...")
        prices_wide = data.pivot_table(
            values=price_col,
            index=data.index,
            columns=ticker_col
        )

        print(f"   Shape: {prices_wide.shape}")
        print(f"   Date range: {prices_wide.index.min().date()} to {prices_wide.index.max().date()}")

        # Calculate from prices
        return self.from_prices(prices_wide, trading_days)

    def from_csv(self, filepath, price_col='Close', ticker_col='Ticker',
                 date_col='Date', trading_days=21):
        """ Load data from CSV and calculate covariance. """
        print(f"📂 Loading data from {filepath}...")
        data = pd.read_csv(filepath)

        return self.from_long_data(data, price_col, ticker_col, date_col, trading_days)

    def save(self, filepath):
        """ Save covariance matrix to CSV. """
        if self.cov_matrix is None:
            raise ValueError("No covariance matrix to save.")

        self.cov_matrix.to_csv(filepath)
        print(f"[*] Saved to {filepath}")

    def load(self, filepath):
        """ Load existing covariance matrix from CSV. """
        self.cov_matrix = pd.read_csv(filepath, index_col=0)

        # Clean duplicates
        self.cov_matrix = self.cov_matrix.loc[
            ~self.cov_matrix.index.duplicated(),
            ~self.cov_matrix.columns.duplicated()
        ]

        print(f"📂 Loaded from {filepath}")
        print(f"   Shape: {self.cov_matrix.shape}")

        return self.cov_matrix

    def get_statistics(self):
        """ Get summary statistics of covariance matrix. """
        if self.cov_matrix is None:
            raise ValueError("No covariance matrix available.")

        variances = np.diag(self.cov_matrix)

        return {
            'n_tickers': len(self.cov_matrix),
            'mean_variance': variances.mean(),
            'min_variance': variances.min(),
            'max_variance': variances.max(),
            'std_variance': variances.std()
        }

    def print_statistics(self):
        """ Print summary statistics. """
        stats = self.get_statistics()

        print(f"\nCovariance Matrix Statistics:")
        print(f"   Tickers: {stats['n_tickers']}")
        print(f"   Mean variance: {stats['mean_variance']:.6f}")
        print(f"   Min variance:  {stats['min_variance']:.6f}")
        print(f"   Max variance:  {stats['max_variance']:.6f}")
        print(f"   Std variance:  {stats['std_variance']:.6f}")