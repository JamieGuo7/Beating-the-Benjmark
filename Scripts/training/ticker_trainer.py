import pandas as pd
import numpy as np
import time

from Scripts.data_pipeline.features import engineer_features
from Scripts.data_pipeline.preprocessing import SequencePreprocessor
from Scripts.models.lstm_model import LSTMPredictor
from Scripts.models.callbacks import create_callbacks
from Scripts.utils.calculate_metrics import calculate_metrics
from Scripts.utils.plotting import plot_predictions

class TickerModelTrainer:
    def __init__(self, feature_cols, target_col, window=30,
                 epochs=100, batch_size=32, learning_rate=0.005,
                 models_dir='../../models', plots_dir='../../plots'):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.models_dir = models_dir
        self.plots_dir = plots_dir

        self.preprocessor = None
        self.model = None

    def prepare_data(self, df_ticker):
        """
        Engineer features and create sequences.
        The function returns the X,y for the model.
        It also returns df_ticker required for forecasting.
        """

        # Remove the numerical index, set 'Date' as index.
        df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
        df_ticker = df_ticker.set_index('Date').sort_index()

        print("[*] Engineering features...")
        df_ticker = engineer_features(df_ticker)

        # Validate
        if len(df_ticker) < 500:
            raise ValueError(f"Insufficient data: {len(df_ticker)} rows")

        df_ticker_truncated = df_ticker.dropna(subset=self.feature_cols + [self.target_col])

        # Create sequences from truncated data (no errors)
        X_raw = df_ticker_truncated[self.feature_cols].values
        y_raw = df_ticker_truncated[self.target_col].values

        print(f"[*] Data points: {len(df_ticker_truncated)}")

        # Create preprocessor and sequences
        self.preprocessor = SequencePreprocessor(window=self.window)
        X, y = self.preprocessor.create_sequences(X_raw, y_raw)

        if len(X) < 100:
            raise ValueError(f"Too few sequences: {len(X)}")

        print(f"[*] Sequences: {X.shape[0]}")

        # df_ticker still has all columns, and still has NaN's
        return X, y, df_ticker

    def split_data(self, X, y, train_ratio=0.80, val_ratio=0.10):
        train_size = int(len(X) * train_ratio)
        val_size = int(len(X) * val_ratio)

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train(self, ticker, df_ticker):
        print(f"\n{'=' * 70}")
        print(f"[>] Training model for: {ticker}")
        print(f"{'=' * 70}")

        # Prepare data
        X, y, df_full = self.prepare_data(df_ticker)

        # Split
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_data(X, y)

        # Preprocess
        X_train_scaled, y_train_scaled = self.preprocessor.fit_transform(X_train, y_train)
        X_val_scaled, y_val_scaled = self.preprocessor.transform(X_val, y_val)
        X_test_scaled, y_test_scaled = self.preprocessor.transform(X_test, y_test)

        # Build model
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        self.model = LSTMPredictor(input_shape, self.learning_rate)

        # Train
        print(f"[>] Training...", end='', flush=True)
        start_time = time.time()

        callbacks = create_callbacks(ticker, self.models_dir)
        history = self.model.train(
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks
        )

        elapsed = time.time() - start_time
        epochs_trained = len(history.history['loss'])
        print(f" Done in {elapsed:.1f}s ({epochs_trained} epochs)")

        # Evaluate
        results = self.evaluate_and_visualise(
            ticker, history,
            X_train_scaled, y_train, y_train_scaled,
            X_val_scaled, y_val, y_val_scaled,
            X_test_scaled, y_test, y_test_scaled,
            df_full
        )

        return results

    def evaluate_and_visualise(self, ticker, history,
                                X_train_scaled, y_train, y_train_scaled,
                                X_val_scaled, y_val, y_val_scaled,
                                X_test_scaled, y_test, y_test_scaled,
                                df_full):
        # Get predictions
        y_train_pred_scaled = self.model.predict(X_train_scaled)
        y_val_pred_scaled = self.model.predict(X_val_scaled)
        y_test_pred_scaled = self.model.predict(X_test_scaled)

        # Inverse transform
        y_train_pred = self.preprocessor.inverse_transform_y(y_train_pred_scaled)
        y_val_pred = self.preprocessor.inverse_transform_y(y_val_pred_scaled)
        y_test_pred = self.preprocessor.inverse_transform_y(y_test_pred_scaled)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)

        print(f"   Train - R²: {train_metrics['r2']:.4f}, Dir Acc: {train_metrics['direction_accuracy']:.2f}%")
        print(f"   Val   - R²: {val_metrics['r2']:.4f}, Dir Acc: {val_metrics['direction_accuracy']:.2f}%")
        print(f"   Test  - R²: {test_metrics['r2']:.4f}, Dir Acc: {test_metrics['direction_accuracy']:.2f}%")

        # Visualise
        print(f"[*] Creating visualisations...")
        plot_predictions(
            ticker,
            y_train, y_train_pred,
            y_val, y_val_pred,
            y_test, y_test_pred,
            history,
            self.plots_dir,
        )

        # Generate forecast
        forecast_pct, forecast_target_date = self.generate_forecast(df_full)

        print(f"   Forecast return: {forecast_pct:.2f}% for {forecast_target_date}")
        return {
            'ticker': ticker,
            'epochs_trained': len(history.history['loss']),

            # Training
            'train_r2': train_metrics['r2'],
            'train_rmse': train_metrics['rmse'],
            'train_dir_acc': train_metrics['direction_accuracy'],

            # Validation
            'val_r2': val_metrics['r2'],
            'val_rmse': val_metrics['rmse'],
            'val_dir_acc': val_metrics['direction_accuracy'],

            # Test
            'test_r2': test_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_dir_acc': test_metrics['direction_accuracy'],

            # Output
            'target_date': forecast_target_date,
            'forecast_return': forecast_pct,
        }

    def generate_forecast(self, df_full):
        df_inference = df_full.tail(self.window).copy()

        if len(df_inference) < self.window:
            return None, None

        # Check for NaNs in FEATURES only (target is allowed to be NaN)
        nan_count = df_inference[self.feature_cols].isna().sum().sum()
        if nan_count > 0:
            print(f"   [!] {nan_count} NaN values in features!")

        X_raw = df_inference[self.feature_cols].values
        X_seq = X_raw.reshape(1, self.window, len(self.feature_cols))
        X_scaled = self.preprocessor.transform(X_seq)

        forecast_scaled = self.model.predict(X_scaled)
        forecast_log = self.preprocessor.inverse_transform_y(forecast_scaled)[0]
        forecast_pct = (np.exp(forecast_log) - 1) * 100

        forecast_target_date = df_inference.index[-1] + pd.offsets.BDay(21)

        return forecast_pct, forecast_target_date.date()


