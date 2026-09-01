import pandas as pd
from pathlib import Path

from trader.config import TARGET_COL
from trader.training.trainer import TickerModelTrainer
from trader.training.batch import BatchTrainer


def load_lstm_data(project_root):
    """ Load preprocessed LSTM-ready data. """
    data_path = project_root / "data" / "raw" / "prices" / "esgu_ohlcv_daily.csv"
    
    df = pd.read_csv(data_path, parse_dates=["Date"])
    return df

def main(max_tickers):
    project_root = Path(__file__).resolve().parents[1]

    test_date_start = '2022-05-02'
    test_date_end = '2023-02-15'
    end_date = '2023-03-18'

    # One Model Training Run == one directory under runs/, named
    # {test_start}_to_{test_end}_v{n}_{family}. Models, plots and results all
    # live inside it, so they cannot drift apart. See docs/adr/0002.
    run_dir = project_root / "runs" / f"{test_date_start}_to_{test_date_end}_v1_lstm"
    models_dir = run_dir / "models"
    plots_dir = run_dir / "plots"
    results_dir = run_dir

    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_lstm_data(project_root)
    all_tickers = sorted(df["Ticker"].unique())

    df = df[df['Date'] <= end_date]

    if max_tickers is not None:
        tickers = all_tickers[:max_tickers]
    else:
        tickers = all_tickers

    print(f"Found {len(all_tickers)} tickers in data. Using {len(tickers)} tickers.")

    target_col = TARGET_COL
    feature_cols = [
        # Long-Term Trend
        'dist_sma200',

        # Momentum
        'ret_21d',
        'momentum_quality',

        # Breakout
        'dist_high52w',

        # Trend Quality
        'efficiency_ratio',
        'adx_slope',

        # Volume
        'vol_ratio',

        # Volatility
        'NATR'
    ]
    
    ticker_trainer = TickerModelTrainer(
        feature_cols=feature_cols,
        target_col= target_col,
        models_dir=str(models_dir),
        plots_dir=str(plots_dir),
    )

    batch_trainer = BatchTrainer(
        ticker_trainer=ticker_trainer,
        results_dir=str(results_dir)
    )

    results_df = batch_trainer.train_all(df, tickers)

    if results_df is None:
        return

    batch_trainer.save_results("results.csv")
    batch_trainer.save_forecasts("forecasts.csv")
    batch_trainer.print_summary()


if __name__ == "__main__":
    main(max_tickers=None)

