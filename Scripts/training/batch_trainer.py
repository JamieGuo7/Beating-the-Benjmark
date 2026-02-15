import pandas as pd
import time
from pathlib import Path


class BatchTrainer:

    def __init__(self, ticker_trainer, results_dir='../../results'):
        self.ticker_trainer = ticker_trainer
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.all_results = []
        self.results_df = None
        self.total_time = 0

    def train_all(self, data, tickers, max_tickers=None):
        if max_tickers:
            tickers = tickers[:max_tickers]

        print(f"\n{'=' * 70}")
        print(f"[!] About to train {len(tickers)} models")
        print(f"   Estimated time: ~{len(tickers) * 0.5:.1f} minutes")
        print(f"{'=' * 70}\n")

        proceed = input("Proceed? (yes/no): ").strip().lower()
        if proceed not in ['yes', 'y']:
            print("Cancelled.")
            return None

        all_results = []
        total_start = time.time()

        for i, ticker in enumerate(tickers, 1):
            print(f"\n{'#' * 70}")
            print(f"Progress: {i}/{len(tickers)} | Elapsed: {(time.time() - total_start) / 60:.1f}m")
            print(f"{'#' * 70}")

            df_ticker = data[data['Ticker'] == ticker].copy()

            try:
                results = self.ticker_trainer.train(ticker, df_ticker)
                if results is not None:
                    all_results.append(results)
            except Exception as e:
                print(f"[X] Error training {ticker}: {str(e)}")
                continue

        self.total_time = (time.time() - total_start) / 60
        self.all_results = all_results

        # Process results
        if len(all_results) == 0:
            print("[X] No tickers successfully trained!")
            return None

        self.results_df = pd.DataFrame(all_results)
        self.results_df = self.results_df.sort_values('test_dir_acc', ascending=False)

        return self.results_df

    def save_results(self, filename='results.csv'):
        if self.results_df is None:
            raise ValueError("No results to save. Run train_all() first.")

        filepath = self.results_dir / filename
        self.results_df.to_csv(filepath, index=False)

        print(f"[*] Results saved to {filepath}")

    def save_forecasts(self, filename='latest_forecasts.csv'):
        if self.results_df is None:
            raise ValueError("No results to save. Run train_all() first.")

        forecasts_df = self.results_df[[
            'ticker', 'target_date', 'forecast_return',
            'test_dir_acc', 'test_r2'
        ]].copy()
        forecasts_df = forecasts_df.dropna(subset=['forecast_return'])

        filepath = self.results_dir / filename
        forecasts_df.to_csv(filepath, index=False)
        
        print(f"[*] Forecasts saved to {filepath}")

    def print_summary(self):
        if self.results_df is None:
            print("No results to summarize.")
            return

        print(f"\n{'=' * 70}")
        print("FINAL SUMMARY")
        print(f"{'=' * 70}")

        print(f"Total tickers trained: {len(self.all_results)}")
        print(f"Total time: {self.total_time:.1f} minutes")
        print(f"Average per ticker: {self.total_time / len(self.all_results):.1f} minutes")

        print(f"\nTest Set Performance:")
        print(f"   Direction Accuracy - Mean: {self.results_df['test_dir_acc'].mean():.2f}%")
        print(f"   Direction Accuracy - Median: {self.results_df['test_dir_acc'].median():.2f}%")
        print(f"   R² - Mean: {self.results_df['test_r2'].mean():.4f}")
        print(f"   R² - Median: {self.results_df['test_r2'].median():.4f}")

        best_ticker = self.results_df.iloc[0]['ticker']
        best_acc = self.results_df['test_dir_acc'].max()
        print(f"   Best Direction Acc: {best_acc:.2f}% ({best_ticker})")

        print(f"\nTop 10 Performers (by test accuracy):")
        display_cols = [
            'ticker', 'test_dir_acc', 'test_r2', 'test_rmse',
            'epochs_trained', 'forecast_return'
        ]
        print(self.results_df[display_cols].head(10).to_string(index=False))

        print(f"\nTraining complete! Check the plots directory for visualizations.")

    def get_top_performers(self, n=10, metric='test_dir_acc'):
        if self.results_df is None:
            raise ValueError("No results available. Run train_all() first.")

        return self.results_df.nlargest(n, metric)

    def get_statistics(self):
        if self.results_df is None:
            raise ValueError("No results available. Run train_all() first.")

        return {
            'n_models': len(self.results_df),
            'total_time_minutes': self.total_time,
            'avg_time_per_ticker': self.total_time / len(self.results_df),
            'mean_dir_acc': self.results_df['test_dir_acc'].mean(),
            'median_dir_acc': self.results_df['test_dir_acc'].median(),
            'mean_r2': self.results_df['test_r2'].mean(),
            'median_r2': self.results_df['test_r2'].median(),
            'best_ticker': self.results_df.iloc[0]['ticker'],
            'best_dir_acc': self.results_df['test_dir_acc'].max()
        }