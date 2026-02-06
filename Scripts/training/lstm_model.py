import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import os
import warnings
import time

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Configuration
DATA_PATH = '../../data/ESGU_LSTM_Ready.csv'
MODELS_DIR = '../../models'
RESULTS_DIR = '../../results'
PLOTS_DIR = '../../plots'

MAX_TICKERS = 300
START_DATE = '2016-03-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
TARGET_COL = '21 Day Forward Return'

# Model Parameters
LEARNING_RATE = 0.005
BATCH_SIZE = 32
WINDOW = 30
EPOCHS = 100

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

# Helper Functions
def engineer_features(df):
    df = df.copy()

    # Momentum
    df['ret_5d'] = df['Close'].pct_change(5)
    df['ret_21d'] = df['Close'].pct_change(21)

    # Volume
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Mean Reversion
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['dist_sma20'] = (df['Close'] / df['sma_20']) - 1

    # RSI Velocity
    df['rsi_vel'] = df['RSI'].diff()

    # Trend Filter
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    df['dist_sma200'] = (df['Close'] / df['sma_200']) - 1

    # Breakout Signal
    df['high_52w'] = df['Close'].rolling(window=252).max()
    df['dist_high52w'] = (df['Close'] / df['high_52w']) - 1

    # EMA Cross
    df['ema_8'] = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['ema_cross'] = (df['ema_8'] / df['ema_21']) - 1

    # Trend Regime
    df['trend_regime'] = (df['Close'] / df['sma_200']) - 1

    # Momentum Quality
    df['momentum_quality'] = (df['Close'] - df['Close'].shift(21)) / df['ATR']

    # ADX Slope
    df['adx_slope'] = df['ADX'].diff(5)

    # Efficiency Ratio
    df['efficiency_ratio'] = (df['Close'] - df['Close'].shift(20)).abs() / \
                             (df['Close'].diff().abs().rolling(20).sum())

    # Forward Return Target
    df['21 Day Forward Return'] = np.log(df['Close'].shift(-21) / df['Close'])

    return df


def create_sequences(X, y, window=30):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# Custom loss function which penalises incorrect direction
def directional_loss(y_true, y_pred):
    huber = tf.keras.losses.Huber()(y_true, y_pred)
    sign_penalty = tf.reduce_mean(
        tf.abs(tf.sign(y_true) - tf.sign(y_pred))
    )
    return huber + 0.5 * sign_penalty


def build_model(input_shape):
    model = Sequential([
        LSTM(units=32, return_sequences=True,
             input_shape=input_shape),
        Dropout(0.2),

        LSTM(units=16, return_sequences=False),
        Dropout(0.2),

        Dense(units=16),
        LeakyReLU(0.1),

        Dense(units=1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=directional_loss,
        metrics=['mae']
    )

    return model


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    # Align lengths
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[-min_len:]
    y_pred = y_pred[-min_len:]

    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    non_zero = (true_sign != 0)
    direction_accuracy = np.mean(true_sign[non_zero] == pred_sign[non_zero]) * 100

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_accuracy': direction_accuracy
    }


def plot_predictions(ticker, y_train, y_train_pred, y_val, y_val_pred,
                     y_test, y_test_pred, history):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Training History (Loss)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'{ticker} - Training History (Loss)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Training History (MAE)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax2.set_title(f'{ticker} - Training History (MAE)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Test Set Predictions vs Actual
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(y_test, label='Actual', linewidth=2, alpha=0.7)
    ax3.plot(y_test_pred, label='Predicted', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Sample', fontsize=12, fontweight='bold')
    ax3.set_ylabel('21-Day Forward Return', fontsize=12, fontweight='bold')
    ax3.set_title(f'{ticker} - Test Set: Predictions vs Actual', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Scatter Plot (Test Set)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(y_test, y_test_pred, alpha=0.5, s=30)

    # Perfect prediction line
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax4.set_xlabel('Actual Returns', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Predicted Returns', fontsize=12, fontweight='bold')
    ax4.set_title(f'{ticker} - Prediction Scatter (Test)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Residuals Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    residuals = y_test - y_test_pred
    ax5.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title(f'{ticker} - Residuals Distribution', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Add overall metrics text
    test_metrics = calculate_metrics(y_test, y_test_pred)
    metrics_text = (
        f"Test Metrics:\n"
        f"R² = {test_metrics['r2']:.4f}\n"
        f"RMSE = {test_metrics['rmse']:.4f}\n"
        f"MAE = {test_metrics['mae']:.4f}\n"
        f"Dir Acc = {test_metrics['direction_accuracy']:.2f}%"
    )

    fig.text(0.98, 0.02, metrics_text, fontsize=11,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save figure
    plt.savefig(f'{PLOTS_DIR}/{ticker}_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

def add_callbacks(ticker):
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        ModelCheckpoint(
            filepath=f'{MODELS_DIR}/lstm_{ticker}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=0
        )
    ]

    return callbacks

def generate_forecast(df_inference, scaler_X, scaler_y, pca, model):
    X_raw = df_inference[feature_cols].values

    # Reshape and process through fitted objects
    X_seq = X_raw.reshape(1, WINDOW, len(feature_cols))
    X_flat = X_seq.reshape(-1, len(feature_cols))

    X_scaled_flat = scaler_X.transform(X_flat)
    X_pca_flat = pca.transform(X_scaled_flat)
    X_scaled = X_pca_flat.reshape(1, WINDOW, -1)

    # Predict and inverse scale
    forecast_scaled = model.predict(X_scaled, verbose=0)
    forecast_log = scaler_y.inverse_transform(forecast_scaled)[0][0]

    return (np.exp(forecast_log) - 1) * 100

def train_ticker_model(ticker, df_ticker):
    print(f"\n{'=' * 70}")
    print(f"🎯 Training model for: {ticker}")
    print(f"{'=' * 70}")

    print("🔧 Engineering features...")
    df_ticker = engineer_features(df_ticker)

    # Check data
    if len(df_ticker) < 500:
        print(f"⚠️  Skipping {ticker}: Insufficient data ({len(df_ticker)} rows)")
        return None

    df_ticker = df_ticker.copy()

    # Create the truncated version for training (Lags by 21 days due to TARGET_COL)
    df_ticker_truncated = df_ticker.dropna(subset=feature_cols + [TARGET_COL])
    print(f"📊 Data points: {len(df_ticker_truncated)}")

    X_raw = df_ticker_truncated[feature_cols].values
    y_raw = df_ticker_truncated[TARGET_COL].values

    X, y = create_sequences(X_raw, y_raw, WINDOW)

    if len(X) < 100:
        print(f"⚠️  Skipping {ticker}: Too few sequences ({len(X)})")
        return None

    print(f"📦 Sequences: {X.shape[0]}")

    # Train/Val/Test split (80/10/10)
    train_size = int(len(X) * 0.80)
    val_size = int(len(X) * 0.10)

    X_train_raw = X[:train_size]
    y_train_raw = y[:train_size]

    X_val_raw = X[train_size:train_size + val_size]
    y_val_raw = y[train_size:train_size + val_size]

    X_test_raw = X[train_size + val_size:]
    y_test_raw = y[train_size + val_size:]

    print(f"   Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

    # Scaling
    # Reshaping as RobustScalar() requires a 2D Array
    # X is a 3D array of shape (number of sequences, window, features)
    N_FEATURES = X_train_raw.shape[2]

    scaler_X = RobustScaler()
    X_train_flat = X_train_raw.reshape(-1, N_FEATURES)
    X_val_flat = X_val_raw.reshape(-1, N_FEATURES)
    X_test_flat = X_test_raw.reshape(-1, N_FEATURES)

    X_train_scaled_flat = scaler_X.fit_transform(X_train_flat)
    X_val_scaled_flat = scaler_X.transform(X_val_flat)
    X_test_scaled_flat = scaler_X.transform(X_test_flat)

    # PCA to reduce noise
    pca = PCA(n_components=0.95)
    X_train_pca_flat = pca.fit_transform(X_train_scaled_flat)
    X_val_pca_flat = pca.transform(X_val_scaled_flat)
    X_test_pca_flat = pca.transform(X_test_scaled_flat)

    N_COMPONENTS = X_train_pca_flat.shape[1]
    print(f"   PCA: {N_FEATURES} → {N_COMPONENTS} components")

    # Reshape back to sequences
    X_train_scaled = X_train_pca_flat.reshape(-1, WINDOW, N_COMPONENTS)
    X_val_scaled = X_val_pca_flat.reshape(-1, WINDOW, N_COMPONENTS)
    X_test_scaled = X_test_pca_flat.reshape(-1, WINDOW, N_COMPONENTS)

    # Scale target
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1))

    model = build_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))
    callbacks = add_callbacks(ticker)

    # Train
    print(f"🚀 Training...", end='', flush=True)
    start_time = time.time()

    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )

    elapsed = time.time() - start_time
    epochs_trained = len(history.history['loss'])
    print(f" Done in {elapsed:.1f}s ({epochs_trained} epochs)")

    # Get predictions for all sets
    y_train_pred_scaled = model.predict(X_train_scaled, verbose=0).flatten()
    y_val_pred_scaled = model.predict(X_val_scaled, verbose=0).flatten()
    y_test_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()

    # Inverse transform
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    # Calculate metrics for all sets
    train_metrics = calculate_metrics(y_train_raw, y_train_pred)
    val_metrics = calculate_metrics(y_val_raw, y_val_pred)
    test_metrics = calculate_metrics(y_test_raw, y_test_pred)

    print(f"   Train - R²: {train_metrics['r2']:.4f}, Dir Acc: {train_metrics['direction_accuracy']:.2f}%")
    print(f"   Val   - R²: {val_metrics['r2']:.4f}, Dir Acc: {val_metrics['direction_accuracy']:.2f}%")
    print(f"   Test  - R²: {test_metrics['r2']:.4f}, Dir Acc: {test_metrics['direction_accuracy']:.2f}%")

    # Create visualization
    print(f"📊 Creating visualizations...")
    try:
        plot_predictions(ticker, y_train_raw, y_train_pred,
                         y_val_raw, y_val_pred,
                         y_test_raw, y_test_pred, history)
        print(f"   ✅ Plot saved: {PLOTS_DIR}/{ticker}_predictions.png")
    except Exception as e:
        print(f"   ⚠️  Error creating plot: {str(e)}")

    # Get the last 30 days of data for the sequence.
    df_inference = df_ticker.tail(WINDOW).copy()
    forecast_pct = generate_forecast(df_inference, scaler_X, scaler_y, pca, model)
    forecast_target_date = df_inference.index[-1] + pd.tseries.offsets.Day(31)

    print(f"   🚀 Forecast return from {df_inference.index[-1].date()} to {forecast_target_date.date()}: {forecast_pct:.2f}%")

    results = {
        'ticker': ticker,
        'epochs_trained': epochs_trained,

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
        'forecast_return': forecast_pct if len(df_inference) >= WINDOW else None
    }

    return results


# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load data
print("📂 Loading data...")
data = pd.read_csv(DATA_PATH)
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(['Ticker', 'Date'])
data.set_index('Date', inplace=True)
print(f"\n🔍 Raw columns in CSV:")
print(data.columns.tolist())

# Check required columns
required_base_cols = ['Close', 'Volume', 'ATR', 'ADX', 'RSI']
missing_base = [col for col in required_base_cols if col not in data.columns]

if missing_base:
    print(f"\n❌ ERROR: Missing required base columns: {missing_base}")
    exit()

print("✅ All required base columns found!")

data = data[(data.index >= START_DATE) & (data.index <= END_DATE)]
print(f"📅 {len(data)} rows starting from {START_DATE} to {END_DATE} found!")

# Get unique tickers, up to MAX_TICKERS
tickers = data['Ticker'].unique()[:MAX_TICKERS]
print(f"📋 Processing {len(tickers)} tickers (limited to {MAX_TICKERS})")

print(f"\n{'=' * 70}")
print(f"⚠️  About to train {len(tickers)} models")
print(f"   Estimated time: ~{len(tickers)/3} minutes")
print(f"   Plots will be saved to: {PLOTS_DIR}/")
print(f"{'=' * 70}")

proceed = input("\nProceed? (yes/no): ").strip().lower()
if proceed not in ['yes', 'y']:
    print("Cancelled.")
    exit()

# Train all tickers
all_results = []
total_start = time.time()

for i, ticker in enumerate(tickers, 1):
    print(f"\n{'#' * 70}")
    print(f"Progress: {i}/{len(tickers)} | Elapsed: {(time.time() - total_start) / 60:.1f}m")
    print(f"{'#' * 70}")

    df_ticker = data[data['Ticker'] == ticker].copy()

    try:
        results = train_ticker_model(ticker, df_ticker)
        if results is not None:
            all_results.append(results)
    except Exception as e:
        print(f"❌ Error training {ticker}: {str(e)}")
        continue

# Get time in minutes
total_time = (time.time() - total_start) / 60

# Save results
print(f"\n{'=' * 70}")
print("📊 FINAL SUMMARY")
print(f"{'=' * 70}")

if len(all_results) == 0:
    print("❌ No tickers successfully trained!")
    exit()

# Sorting the results by test direction accuracy
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('test_dir_acc', ascending=False)

# Save results
results_df.to_csv(f'{RESULTS_DIR}/results.csv', index=False)

# Save forecasts (simple version of results)
forecasts_df = results_df[['ticker', 'target_date', 'forecast_return',
                           'test_dir_acc', 'test_r2']].copy()
forecasts_df = forecasts_df.dropna(subset=['forecast_return'])
forecasts_df.to_csv(f'{RESULTS_DIR}/latest_forecasts.csv', index=False)

print(f"Total tickers trained: {len(all_results)}")
print(f"Total time: {total_time:.1f} minutes")
print(f"Average per ticker: {total_time / len(all_results):.1f} minutes")

print(f"\n📊 Test Set Performance:")
print(f"   Direction Accuracy - Mean: {results_df['test_dir_acc'].mean():.2f}%")
print(f"   Direction Accuracy - Median: {results_df['test_dir_acc'].median():.2f}%")
print(f"   R² - Mean: {results_df['test_r2'].mean():.4f}")
print(f"   R² - Median: {results_df['test_r2'].median():.4f}")
print(f"   Best Direction Acc: {results_df['test_dir_acc'].max():.2f}% ({results_df.iloc[0]['ticker']})")

print(f"\n💾 Results saved to:")
print(f"   {RESULTS_DIR}/results.csv")
print(f"   {RESULTS_DIR}/latest_forecasts.csv")
print(f"   {MODELS_DIR}/lstm_{{ticker}}_best.keras")
print(f"   {PLOTS_DIR}/{{ticker}}_predictions.png")

# Display top 10 (we sorted previously by test direction accuracy)
print(f"\n🏆 Top 10 Performers (by test accuracy):")
display_cols = ['ticker', 'test_dir_acc', 'test_r2', 'test_rmse',
                'epochs_trained', 'forecast_return']
print(results_df[display_cols].head(10).to_string(index=False))

print(f"\n🎉 Training complete! Check the plots directory for visualizations.")
