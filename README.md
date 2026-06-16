# 🐟 Beating the Benjmark

Beating the Benjmark is a Warwick AI Society project undertaken by [Jamie Guo](https://github.com/JamieGuo7) and [Aaron Moorby](https://github.com/Aaronm3103) which explores combining AI Techniques and traditional portfolio theory to construct an optimal portfolio for a one-month timeline. 

The goal of the project was to outperform the Warwick AI Society Pet Fish, Benji:
- Benji makes trades by swimming, swimming to the left of the tank is a sell order, whilst swimming to the right is a buy order.
- Inspired by Michael Reeves Viral Video: https://www.youtube.com/watch?v=USKD3vPD6ZA

We won this competition and made a 6.57% return on the month compared to Benji's 0.45%.

## 🧠 Methodology
1. **Grab enough data**  
   - The model uses 12 years of daily financial data from yfinance to capture meaningful market patterns.  
   - Fun fact: Jamie got banned from yfinance for requesting too much data at one point.
2. **Extract and process signals**  
   - Key features include:  
     - `dist_sma200` – tells the model if an asset is generally trending up or down (long-term trend).  
     - `efficiency_ratio` – measures how reliable a trend is, helping the model ignore noise.  
   - Features were scaled using a **RobustScaler** to reduce the impact of outliers.  
3. **Use an LSTM to forecast returns**  
   - Two-layer LSTM predicts one-month returns based on the processed features.
   - A **custom loss function** punishes the model for incorrect direction.
4. **Black-Litterman Portfolio Optimisation**  
   - Expected returns and their RMSE values are fed into a **Black-Litterman framework**.  
   - Produces a **risk-adjusted portfolio** that accounts for uncertainty in the LSTM predictions.
