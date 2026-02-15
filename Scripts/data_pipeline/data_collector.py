import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class DataCollector:
    """ Handles data acquisition and basic cleaning. """

    def __init__(self, ticker_path, file_path = None, period = '12y', interval = '1d'):
        self.ticker_path = ticker_path
        self.file_path = file_path

        self.period = period
        self.interval = interval

        self.tickers = self.load_tickers()

        if self.file_path:
            print(f"Loading existing data from {self.file_path}...")
            self.clean_data = pd.read_csv(self.file_path)
            self.clean_data['Date'] = pd.to_datetime(self.clean_data['Date'])
        else:
            self.raw_data = self.fetch(period = self.period, interval = self.interval)
            self.clean_data = self.tidy(self.raw_data)

    def get_clean_data(self):
        return self.clean_data

    def load_tickers(self):
        """ Assumes that tickers are in a text file, one after the other. """
        with open(self.ticker_path, 'r') as file:
            return [line.strip() for line in file]

    def fetch(self, start = None, end = None, period = None, interval = None):
        """ Fetches data from Yahoo Finance. """

        # Use passed arguments if they exist
        fetch_period = period if period else self.period
        fetch_interval = interval if interval else self.interval

        # A start date dictates period, so then it must be none for yfinance
        if start:
            fetch_period = None

        print(f"Downloading data for {len(self.tickers)} tickers...")
        raw_data = yf.download(
                    self.tickers,
                    start = start,
                    end = end,
                    period = fetch_period,
                    interval = fetch_interval,
                    group_by = 'column',
                    auto_adjust=True)
        return raw_data


    def tidy(self, df):
        """ Transform multi-index data into a tidy format"""
        if df.empty:
            raise ValueError("No raw data. Call fetch() first.")
            return

        tidy_data = df.stack(level=1, future_stack=True).reset_index()
        tidy_data.rename(columns={'level_1': 'Ticker'}, inplace=True)
        return tidy_data

    def append_data(self):
        """ Appends data to existing dataframe """
        if self.clean_data is None or self.clean_data.empty:
            print("No existing data to append to.")
            return

        last_date = self.clean_data['Date'].max()
        start_date = (last_date + timedelta(days = 1)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')

        if start_date >= today:
            print(f"Data is already up to date (Last entry: {last_date.date()}).")
            return

        print(f"Fetching new data from {start_date} to {today}...")
        new_raw = self.fetch(start = start_date, end = today)

        if not new_raw.empty:
            new_clean = self.tidy(new_raw)
            self.clean_data = pd.concat([self.clean_data, new_clean], ignore_index=True)
            self.clean_data = self.clean_data.drop_duplicates(subset = ['Date', 'Ticker'])
            print(f"Successfully appended {len(new_clean)} rows.")
        else:
            print("No new data found on Yahoo Finance for those dates")

    def save_data(self, path = None):
        target_path = path or self.file_path

        if target_path:
            self.clean_data.to_csv(target_path, index = False)
            print(f"Data saved to {target_path}")

project_root = Path(__file__).resolve().parents[2]

collector = DataCollector(
    ticker_path = str(project_root / "data" / "ESGU_Tickers.txt"),
    file_path = str(project_root / "data" / "ESGU_LSTM_Ready.csv"),
    period = '12y',
    interval = '1d'
)

collector.append_data()

data = collector.get_clean_data()
collector.save_data()

