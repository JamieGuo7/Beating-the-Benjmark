import numpy as np
import ta

def engineer_features(df):
    """ Engineer technical indicators and features. """
    df = df.copy()

    # Trend
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    df['dist_sma200'] = (df['Close'] / df['sma_200']) - 1

    # Momentum
    df['ret_21d'] = df['Close'].pct_change(21)
    df['momentum_quality'] = (df['Close'] - df['Close'].shift(21)) / df['ATR']

    # Breakout Signal
    df['high_52w'] = df['Close'].rolling(window=252).max()
    df['dist_high52w'] = (df['Close'] / df['high_52w']) - 1

    # Efficiency Ratio
    df['efficiency_ratio'] = (
            (df['Close'] - df['Close'].shift(20)).abs() /
            (df['Close'].diff().abs().rolling(20).sum())
    )

    # ADX Slope
    df['ADX'] = ta.trend.adx(df['High'], df['Low'],
                             df['Close'])
    df['adx_slope'] = df['ADX'].diff(5)

    # Volume
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Volatility
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'],
                                                 df['Close'])
    df['NATR'] = (df['ATR'] / df['Close']) * 100

    # Forward Return Target
    df['21 Day Forward Return'] = np.log(df['Close'].shift(-21) / df['Close'])

    return df