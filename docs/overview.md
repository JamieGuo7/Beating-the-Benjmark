# Overview

## Motivation
This project arose from trying to win the Warwick AI trading competition. This is centred on who could create the best portfolio to hold across one month, which beats out the trading strategy of a fish. In particular where the fish can buy or sell stocks based on its movements. 

We wanted to build a pipeline that takes data, applies feature engineering and then train a model to generate forecasts. From those forecasts we can then find an optimal portfolio to hold.

## Design

There are two main design decisions:

1. Forecast - How do we generate a reliable forecast? 
2. Portfolio - With this forecast information, what portfolio should we choose?

These design decisions break down to many other design decisions. However, if we get these questions right, then, in theory, we should have an optimal portfolio. In that, the portfolio we choose will be the best for the available forecast, and our available forecasts are as accurate as can be. 

## Our model in a snapshot

```
ESGU tickers  →  daily OHLCV  →  8 technical features
                                          ↓
                          one LSTM per ticker (274 of them)
                                          ↓
                       Forecast: 21-trading-day forward return
                                          ↓
              Black-Litterman  ←  covariance matrix + market caps
                                          ↓
                       Efficient Frontier → portfolio weights
                                          ↓
                            Backtest vs the ESGU benchmark
```

We use the [ESGU](https://www.ishares.com/us/products/286007/), index which contains 275 US equities. Every ticker gets its own model rather than one model across the index.

## Where to go next

- [`data.md`](data.md)
- [`pipeline.md`](pipeline.md)
- [`modelling.md`](modelling.md)
