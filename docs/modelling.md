# Modelling

This page is a more in depth exploration of the forecast stages in [`pipeline.md`](pipeline.md),
and talks about the decisions that we make and justification.

## The target

```python
df['21 Day Forward Return'] = np.log(df['Close'].shift(-21) / df['Close'])
```
**Why 21 days.** One trading month, matching the challenge's target horizon.
Short enough to support a monthly rebalance, long enough that a single day's
noise doesn't dominate. Daily returns are almost pure noise at this
signal-to-noise ratio and annual returns would leave only ~11 non-overlapping
observations per ticker.

**Why a log return.** Additive across time - the sum of 21 daily log returns is
the 21-day log return, which simple returns do not give you. Roughly symmetric
around zero, which suits a regression loss. Better-behaved tails.

**What this costs.** A 21-day forward return sampled daily means consecutive
labels share 20 of their 21 days. The observations are not independent -
adjacent rows are almost the same label shifted by one day, like a moving
average of itself.

This has consequences:
- **Effective sample size is much smaller than N.** With ~20/21 overlap,
  a year of daily-sampled labels behaves more like ~12 independent
  observations, not ~250. Any variance or significance estimate that assumes
  i.i.d. rows will be overconfident.
- **Standard cross-validation leaks.** A random train/test split puts
  near-duplicate labels on both sides of the split, so validation error
  understates true generalisation error. This is why splits need to be
  time-blocked (and, per the earlier limitation, why the 30-day sequence
  window still leaks a bit even then).
- **Naive residual diagnostics look better than they are.** Autocorrelated
  residuals can pass eyeball checks or even some statistical tests while
  hiding a model that's just tracking yesterday's label.

The usual mitigations are to report metrics on non-overlapping windows (or
explicitly account for the overlap, e.g. Newey-West-style standard errors),
and to make sure the train/test split respects time blocks rather than
random shuffling.

## The features

Eight, from [`trader/data/features.py`](../trader/data/features.py). We have stationary predictors.

| Feature | Definition | What it is for |
|---|---|---|
| `NATR` | `ATR / Close × 100` | Volatility, as a % of price. Comparable across a stocks. |
| `dist_sma200` | `Close / SMA200 − 1` | Long-term trend - how far above/below the 200-day mean. |
| `ret_21d` | `Close.pct_change(21)` | Momentum over  the forecast horizon. |
| `momentum_quality` | `(Close − Close[−21]) / ATR` | Momentum relative to volatility.. |
| `dist_high52w` | `Close / 252-day high − 1` | Breakout / drawdown position. Zero means at a 52-week high. |
| `efficiency_ratio` | `\|Δ20\| / Σ\|daily Δ\|` over 20d | Trend quality: 1.0 = straight line, near 0 = chop. Distinguishes real trend from noise. |
| `adx_slope` | `ADX.diff(5)` | Is trend strength building or fading? |
| `vol_ratio` | `Volume / 20-day mean Volume` | Is this move backed by volume? |

Indicators come from [`ta`](https://technical-analysis-library-in-python.readthedocs.io/).

These features are indeed far from finished, and it is not known if they are the most predictive. We will do a feature exploration update to this project, which will hopefully yield more predictive features and consider segmenting data into sectors. 

## Preprocessing

[`trader/data/preprocessing.py`](../trader/data/preprocessing.py) -
`SequencePreprocessor`

Order of operations: window → scale → PCA.

1. **Windowing.** 30 consecutive days of the 8 features → one sample of shape
   `(30, 8)`. Windows overlap by 29 days.
2. **`RobustScaler`.** Centres on the median and scales by IQR, so one
   earnings-day spike doesn't compress everything else into a sliver of the
   range. Standard scaling on financial data gets dominated by outliers.
3. **PCA at 95% variance.** The eight features are correlated by
   construction - `ret_21d` and `momentum_quality` share a numerator, 
   `dist_sma200` and `dist_high52w` both measure position within a range.
   PCA decorrelates them and reduces to less noise.

The scalers and PCA are fitted on train only, then pickled per ticker alongside the model - so a backtest replays the exact transform instead of re-deriving one. That's why a Model Set (model + fitted preprocessors) is the unit inference loads.

Two known issues:

- Windows are built before the split, so a window straddling the train/val boundary leaks rows from both. Left alone for now so the next retrain isolates one change at a time.
- Train→test drift isn't handled. Since the scaler and PCA are fit on train alone, any shift in the test-set distribution falls partly outside the fitted range. How much it shifts isn't measured (see [`data.md`](data.md)).

## The model

[`trader/models/lstm.py`](../trader/models/lstm.py) - `LSTMPredictor`

```
Input (30, n_components)
LSTM 32, return_sequences=True    → Dropout 0.2
LSTM 16                           → Dropout 0.2
Dense 16 → LeakyReLU(0.1)
Dense 1
```
Adam, lr 0.005, batch 32, up to 100 epochs. Callbacks ([`callbacks.py`](callbacks.py)): EarlyStopping (patience 15, best-weight restore), ModelCheckpoint on best epoch, ReduceLROnPlateau (halve after 5 flat epochs, floor 1e-5).

**Why an LSTM.** Order carries information - a stock that fell then recovered isn't the same as one that rose then fell, same endpoints or not.

**Why it's small.** 32 → 16 units is tiny by deep-learning standards since ~2,200 training sequences per model against a low-signal target. When using larger number of units the model memorised noise and had poor test performance.

## The loss function

This is the most interesting piece of the model.

The intuition: for a portfolio, getting the direction right matters more than
getting the magnitude right. Predicting +3% when the truth is +1% costs little.
Whilst predicting −1% when the truth is +1% puts you on the wrong side of the trade.
Plain MSE treats those as similar-sized errors.

So the loss is Huber (robust to the fat tails in returns) plus a penalty for
sign disagreement:

```python
def directional_loss(y_true, y_pred, penalty_weight=0.1):
    huber = tf.keras.losses.Huber()(y_true, y_pred)
    direction_error = tf.maximum(0.0, -y_true * y_pred)
    return huber + (penalty_weight * tf.reduce_mean(direction_error))
```

`-y_true * y_pred` is positive exactly when the signs disagree, and its magnitude
scales with how confidently wrong the prediction was. `tf.maximum(0.0, ·)` makes
it a hinge - zero cost when the sign is right.

**The version this replaced was broken:**

```python
sign_penalty = tf.reduce_mean(tf.abs(tf.sign(y_true) - tf.sign(y_pred)))
return huber + 0.5 * sign_penalty
```

`tf.sign` is a step function. Its gradient is zero almost everywhere, so this
term contributed **nothing** to training. Therefore the models were effectively trained on
plain Huber loss while appearing to optimise for direction. The fix is the same
idea expressed differently where the hinge is differentiable wherever it is nonzero.

**Every model currently in `runs/` was trained with this broken version.** Any
retrain is therefore not comparable to the existing results, which is why the
existing numbers have to be captured as a baseline before anything is retrained.

> Anything loading a saved `.keras` file must pass
 `custom_objects={'directional_loss': directional_loss}` or Keras cannot
> reconstruct the model.

## One model per ticker

274 models, one per stock, rather than one model across the index.

**Problems with this approach**

- Each model gets ~2,200 training sequences. A shared model gets ~600,000.
- It costs ~2 hours per full run, making model iteration tricky.
- Assumes returns are independent of other stocks. Stock returns aren't independent - sector rotations, market regime, common risk factors all carry signal that only shows up when you look across tickers together. A per-ticker model can't see other stocks history and that is a limiting factor. 

## Per-ticker diagnostics

Every trained model produces a train/val/test prediction plot and a loss curve
([`trader/evaluation/plots.py`](../trader/evaluation/plots.py)), saved to
`runs/{run}/plots/{TICKER}_predictions.png`.