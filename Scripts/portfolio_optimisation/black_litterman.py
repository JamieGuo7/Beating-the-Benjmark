import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import black_litterman, BlackLittermanModel, EfficientFrontier, risk_models
from pypfopt.expected_returns import mean_historical_return

class PortfolioOptimiser:
    def __init__(self, cov_matrix_path, risk_aversion = 2.5, tau = 0.025):
        self.cov_df = pd.read_csv(cov_matrix_path, index_col=0)
        self.cov_df = self.cov_df.loc[~self.cov_df.index.duplicated(), ~self.cov_df.columns.duplicated()]
        self.risk_aversion = risk_aversion
        self.tau = tau

        # Store last optimisation results
        self.last_weights = None
        self.last_performance = None
        self.last_bl_model = None
        self.last_ef = None

    def _get_market_caps(self, tickers):
        print(f"[*] Fetching market caps for {len(tickers)} tickers...")
        tickers_obj = yf.Tickers(" ".join(STOCKS))
        return pd.Series({t: tickers_obj.tickers[t].info.get('marketCap', 1e9) for t in STOCKS})

    def align_data(self, forecast_df):
        """
        Align forecast data to be in the same order as the covariance matrix
        """
        # Ensure forecast_df is indexed by ticker
        if 'ticker' in forecast_df.columns:
            forecast_df = forecast_df.set_index('ticker')

        # Find common tickers
        tickers = forecast_df.index.intersection(self.cov_df.index).tolist()

        if len(tickers) == 0:
            raise ValueError("No common tickers between forecasts and covariance matrix")

        print(f"[ok] Aligned {len(tickers)} tickers")

        # Align both dataframes
        S = self.cov_df.loc[tickers, tickers]
        df = forecast_df.loc[tickers]

        return df, S, tickers

    def calculate_omega(self, S, rmses, tickers):
        """
        Calculate omega (view uncertainty matrix) based on model RMSEs.
        """
        view_variances = np.diag(S)
        confidence_multiplier = (rmses / rmses.mean()) ** 2
        omega = np.diag(self.tau * view_variances * confidence_multiplier)

        return omega

    def optimise_black_litterman(self, forecast_df, method='max_sharpe'):
        """
        Perform Black-Litterman optimisation combining forecasts with market priors.
        """

        print(f"\n{'=' * 70}")
        print(f"BLACK-LITTERMAN PORTFOLIO OPTIMISATION")
        print(f"{'=' * 70}")

        # 1. Align data
        df, S, tickers = self._align_data(forecast_df)

        # 2. Extract views and uncertainties
        views = (df['forecast_return'] / 100).to_dict()
        rmses = df['test_rmse'].values

        # 3. Get market equilibrium (prior)
        mcaps = self._get_market_caps(tickers)
        pi = black_litterman.market_implied_prior_returns(
            mcaps,
            self.risk_aversion,
            S
        )

        # 4. Calculate view uncertainty (omega)
        omega = self.calculate_omega(S, rmses, tickers)

        # 5. Black-Litterman model
        bl = BlackLittermanModel(
            S,
            pi=pi,
            absolute_views=views,
            omega=omega,
            tau=self.tau
        )

        # Store for diagnostics
        self.last_bl_model = bl

        # 6. Get posterior returns and covariance
        ret_bl = bl.bl_returns()
        S_bl = bl.bl_cov()

        # 7. optimise portfolio
        ef = EfficientFrontier(ret_bl, S_bl)
        ef.add_constraint(lambda w: w >= 0)  # No shorting

        # Choose optimisation method
        if method == 'max_sharpe':
            ef.max_sharpe()
        elif method == 'min_volatility':
            ef.min_volatility()
        elif method == 'max_quadratic_utility':
            ef.max_quadratic_utility(risk_aversion=self.risk_aversion)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.last_ef = ef

        # 8. Get results
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)

        self.last_weights = weights
        self.last_performance = performance

        # 9. Generate diagnostics
        diagnostics = self.generate_diagnostics(df, S, rmses, omega, ret_bl)

        # 10. Print report
        self.print_report(weights, performance, diagnostics, df)

        return {
            'weights': weights,
            'performance': performance,
            'bl_returns': ret_bl,
            'diagnostics': diagnostics
        }

    def optimise_markowitz(self, returns_df, method='tangency'):
        print(f"\n{'=' * 70}")
        print(f"MARKOWITZ PORTFOLIO OPTIMISATION ({method.upper()})")
        print(f"{'=' * 70}")

        # Calculate expected returns and covariance
        mu = mean_historical_return(returns_df)
        S = risk_models.sample_cov(returns_df)

        # optimise
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0)  # No shorting

        if method == 'tangency':
            ef.max_sharpe()
        elif method == 'min_volatility':
            ef.min_volatility()
        elif method == 'efficient_risk':
            target_volatility = 0.15  # 15% annualized
            ef.efficient_risk(target_volatility)
        else:
            raise ValueError(f"Unknown method: {method}")

        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)

        self.last_weights = weights
        self.last_performance = performance

        # Print results
        print("\n--- Optimised Portfolio Weights ---")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for ticker, weight in sorted_weights:
            if weight > 0.0001:
                print(f"  {ticker:6}: {weight:7.2%}")

        print(f"\n--- Expected Portfolio Performance ---")
        print(f"  Expected Annual Return: {performance[0]:.2%}")
        print(f"  Annual Volatility:      {performance[1]:.2%}")
        print(f"  Sharpe Ratio:           {performance[2]:.3f}")
        print(f"{'=' * 70}\n")

        return {
            'weights': weights,
            'performance': performance
        }

    def generate_diagnostics(self, df, S, rmses, omega, ret_bl):
        """
        Generate diagnostic information about the optimisation.
        """
        omega_diag = np.diag(omega)
        prior_cov_diag = np.diag(self.tau * S)
        monthly_vols = np.sqrt(np.diag(S))
        rmse_to_vol = rmses / monthly_vols

        return {
            'tau': self.tau,
            'omega_min': omega_diag.min(),
            'omega_max': omega_diag.max(),
            'omega_mean': omega_diag.mean(),
            'uncertainty_ratio': omega_diag.mean() / prior_cov_diag.mean(),
            'rmse_min': rmses.min(),
            'rmse_max': rmses.max(),
            'rmse_mean': rmses.mean(),
            'vol_min': monthly_vols.min(),
            'vol_max': monthly_vols.max(),
            'vol_mean': monthly_vols.mean(),
            'rmse_vol_ratio_mean': rmse_to_vol.mean(),
            'posterior_return_mean': ret_bl.mean(),
            'posterior_return_min': ret_bl.min(),
            'posterior_return_max': ret_bl.max(),
            'top_confident': df.nsmallest(3, 'test_rmse')[['forecast_return', 'test_rmse']].to_dict('index'),
            'least_confident': df.nlargest(3, 'test_rmse')[['forecast_return', 'test_rmse']].to_dict('index')
        }

    def print_report(self, weights, performance, diagnostics, forecast_df):
        """
        Print comprehensive optimisation report.
        """
        print("\n--- BLACK-LITTERMAN DIAGNOSTICS ---")
        print(f"Tau: {diagnostics['tau']}")
        print(f"\nOmega (View Uncertainty):")
        print(f"  Min:  {diagnostics['omega_min']:.6f}")
        print(f"  Max:  {diagnostics['omega_max']:.6f}")
        print(f"  Mean: {diagnostics['omega_mean']:.6f}")

        print(f"\nUncertainty Ratio (View/Prior): {diagnostics['uncertainty_ratio']:.2f}")
        print("  (Ideal: 0.1 to 10 for balanced influence)")

        print(f"\nModel Accuracy vs Market Volatility:")
        print(f"  Mean RMSE/Vol Ratio: {diagnostics['rmse_vol_ratio_mean']:.2f}")
        print("  (Ratios < 1.0 suggest predictions better than random)")

        print(f"\nTop 3 Most Confident Predictions (Lowest RMSE):")
        for ticker, metrics in diagnostics['top_confident'].items():
            print(f"  {ticker:6}: Forecast {metrics['forecast_return']:6.2f}% | RMSE {metrics['test_rmse']:.4f}")

        print(f"\nTop 3 Least Confident Predictions (Highest RMSE):")
        for ticker, metrics in diagnostics['least_confident'].items():
            print(f"  {ticker:6}: Forecast {metrics['forecast_return']:6.2f}% | RMSE {metrics['test_rmse']:.4f}")

        print("\n--- BLACK-LITTERMAN POSTERIOR RETURNS ---")
        print(f"  Mean: {diagnostics['posterior_return_mean']:.4f}")
        print(f"  Min:  {diagnostics['posterior_return_min']:.4f}")
        print(f"  Max:  {diagnostics['posterior_return_max']:.4f}")

        print("\n--- OPTIMISED PORTFOLIO WEIGHTS ---")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for ticker, weight in sorted_weights:
            if weight > 0.0001:
                forecast = forecast_df.loc[ticker, 'forecast_return']
                print(f"  {ticker:6}: {weight:7.2%}  (Forecast: {forecast:6.2f}%)")

        print(f"\n--- EXPECTED PORTFOLIO PERFORMANCE ---")
        print(f"  Expected Monthly Return: {performance[0]:.2%}")
        print(f"  Monthly Volatility:      {performance[1]:.2%}")
        print(f"  Sharpe Ratio:            {performance[2]:.3f}")
        print(f"{'=' * 70}\n")

    def save_weights(self, filepath):
        """
        Save portfolio weights to CSV.
        """
        if self.last_weights is None:
            raise ValueError("No weights to save. Run Optimisation first.")

        weights_df = pd.DataFrame.from_dict(
            self.last_weights,
            orient='index',
            columns=['weight']
        )
        weights_df = weights_df[weights_df['weight'] > 0.0001].sort_values('weight', ascending=False)
        weights_df.to_csv(filepath)
        print(f"[*] Weights saved to {filepath}")


    
# Getting data
STOCKS = []
with open('../../data/ESGU_Tickers.txt', 'r') as file:
    for line in file:
        STOCKS.append(line.strip())

# UNCOMMENT for our selection of stocks
# STOCKS = ["CEG", "KEYS", "LRCX", "HWM", "CARR", "CCK", "EME", "BKNG", "ES",
#           "FBIN", "NVDA", "GOOG", "APD", "CNH", "MCD", "TT", "MRK", "CBOE", "KDP", "EA",
#           "MCK", "AKAM", "NOC", "AMGN", "SBUX", "PHM", "CRWD", "CAT", "HUBB", "WELL", "BAC",
#           "NDAQ", "AAPL"]

# Data Cleaning
results_df = pd.read_csv('../../results/results.csv')
results_df = results_df.drop_duplicates(subset='ticker')
results_df = results_df[results_df['ticker'].isin(STOCKS)].set_index('ticker')
results_df = results_df.dropna(subset=['forecast_return', 'test_rmse'])

# Re-index to match the order of our stocks list exactly
results_df = results_df.reindex(STOCKS)


# Load Covariance and dropping duplicates
cov_df = pd.read_csv('../../data/market_covariance_matrix.csv', index_col=0)
cov_df = cov_df.loc[~cov_df.index.duplicated(), ~cov_df.columns.duplicated()]

# Only keep stocks for which we have covariances
STOCKS = results_df.index.intersection(cov_df.index).tolist()

# Align indices
S = cov_df.loc[STOCKS, STOCKS]
results_df = results_df.loc[STOCKS]

# Extract Views (Q) and RMSEs for Confidence (Omega)
lstm_views = (results_df['forecast_return'] / 100).to_dict()
rmses = results_df['test_rmse'].values

print("Fetching market caps...")
tickers_obj = yf.Tickers(" ".join(STOCKS))
# Convert market caps to a Series aligned with our Covariance Matrix index
mcaps = pd.Series({t: tickers_obj.tickers[t].info.get('marketCap', 1e9) for t in STOCKS})


pi = black_litterman.market_implied_prior_returns(
    mcaps,
    risk_aversion=2.5,
    cov_matrix=S
)

# We use the squared RMSE to tell the model how much to trust each individual stock's forecast.
tau = 0.025
P = np.eye(len(STOCKS))
view_variances = np.diag(S)
confidence_multiplier = (rmses / rmses.mean())**2
omega = np.diag(tau * view_variances * confidence_multiplier)

# Diagnostics to evaluate model performance
print("\n--- Black-Litterman Diagnostics ---")
print(f"Tau: {tau}")
print(f"\nOmega diagonal values (view uncertainties):")
print(f"  Min: {np.min(np.diag(omega)):.6f}")
print(f"  Max: {np.max(np.diag(omega)):.6f}")
print(f"  Mean: {np.mean(np.diag(omega)):.6f}")

print(f"\nPrior covariance (tau * S) diagonal:")
print(f"  Mean: {np.mean(np.diag(tau * S)):.6f}")

ratio = np.mean(np.diag(omega)) / np.mean(np.diag(tau * S))
print(f"\nRatio of view uncertainty to prior uncertainty: {ratio:.2f}")
print("  (Should be roughly 0.1 to 10 for balanced influence)")

print(f"\nRMSE Statistics:")
print(f"  Min: {rmses.min():.4f}")
print(f"  Max: {rmses.max():.4f}")
print(f"  Mean: {rmses.mean():.4f}")

print(f"\nMonthly Volatility Statistics:")
monthly_vols = np.sqrt(np.diag(S))
print(f"  Min: {monthly_vols.min():.4f}")
print(f"  Max: {monthly_vols.max():.4f}")
print(f"  Mean: {monthly_vols.mean():.4f}")

# Check RMSE vs Volatility
print(f"\nRMSE/Volatility Ratios (forecast accuracy check):")
rmse_to_vol = rmses / monthly_vols
print(f"  Min: {rmse_to_vol.min():.2f}")
print(f"  Max: {rmse_to_vol.max():.2f}")
print(f"  Mean: {rmse_to_vol.mean():.2f}")
print("  (Ratios < 1.0 suggest predictions better than random)")

# Show which stocks have best/worst predictions
print(f"\nTop 5 Most Confident Predictions (lowest RMSE):")
top_confident = results_df.nsmallest(5, 'test_rmse')[['forecast_return', 'test_rmse']]
for ticker, row in top_confident.iterrows():
    print(f"  {ticker}: Forecast={row['forecast_return']:.2f}%, RMSE={row['test_rmse']:.4f}")

print(f"\nTop 5 Least Confident Predictions (highest RMSE):")
least_confident = results_df.nlargest(5, 'test_rmse')[['forecast_return', 'test_rmse']]
for ticker, row in least_confident.iterrows():
    print(f"  {ticker}: Forecast={row['forecast_return']:.2f}%, RMSE={row['test_rmse']:.4f}")

bl = BlackLittermanModel(
    S,
    pi=pi,
    absolute_views=lstm_views,
    omega=omega,
    tau=tau
)

# Combined Returns and Covariance
ret_bl = bl.bl_returns()
S_bl = bl.bl_cov()

print("\n--- Black-Litterman Posterior Returns ---")
print(f"Mean posterior return: {ret_bl.mean():.4f}")
print(f"Min posterior return: {ret_bl.min():.4f}")
print(f"Max posterior return: {ret_bl.max():.4f}")

# Optimisation
ef = EfficientFrontier(ret_bl, S_bl)
ef.add_constraint(lambda w: w >= 0) # No short selling

try:
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    print("\n--- Optimised Portfolio Weights ---")
    sorted_weights = sorted(cleaned_weights.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_weights:
        if weight > 0:
            print(f"{ticker}: {weight:.2%}")

    # Portfolio performance
    perf = ef.portfolio_performance(verbose=False)
    print(f"\n--- Expected Portfolio Performance ---")
    print(f"Expected Monthly Return: {perf[0]:.2%}")
    print(f"Monthly Volatility: {perf[1]:.2%}")
    print(f"Sharpe Ratio: {perf[2]:.3f}")

except Exception as e:
    print(f"Optimisation failed: {e}. Defaulting to Min Volatility.")
    weights = ef.min_volatility()
    print(ef.clean_weights())