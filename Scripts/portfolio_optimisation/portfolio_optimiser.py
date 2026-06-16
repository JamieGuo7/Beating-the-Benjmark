import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import black_litterman, BlackLittermanModel, EfficientFrontier, risk_models
from pypfopt.expected_returns import mean_historical_return

from Scripts.data_pipeline.covariance_calculator import CovarianceCalculator


class PortfolioOptimiser:
    def __init__(self, cov_df, risk_aversion = 2.5, tau = 0.025):
        self.cov_df = cov_df
        self.cov_df = self.cov_df.loc[~self.cov_df.index.duplicated(), ~self.cov_df.columns.duplicated()]
        self.risk_aversion = risk_aversion
        self.tau = tau

        # Store last optimisation results
        self.last_weights = None
        self.last_performance = None
        self.last_bl_model = None
        self.last_ef = None

    def get_market_caps(self, tickers):
        print(f"[*] Fetching market caps for {len(tickers)} tickers...")
        tickers_obj = yf.Tickers(" ".join(tickers))
        return pd.Series({t: tickers_obj.tickers[t].info.get('marketCap', 1e9) for t in tickers})

    def align_data(self, forecast_df):
        """ Align forecast data to be in the same order as the covariance matrix """

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
        """ Calculate omega (view uncertainty matrix) based on model RMSEs. """

        view_variances = np.diag(S)
        confidence_multiplier = (rmses / rmses.mean()) ** 2
        omega = np.diag(self.tau * view_variances * confidence_multiplier)

        return omega

    def optimise_black_litterman(self, tickers_raw, forecast_df, method='max_sharpe'):
        """
        Perform Black-Litterman optimisation combining forecasts with market priors.
        Assumes that forecast_df has 'forecast_return' and 'test_rmse' columns.

        Tickers_raw is those tickers in the forecast_df.
        This could differ from tickers returned by align_data.
        """

        print(f"\n{'=' * 70}")
        print(f"BLACK-LITTERMAN PORTFOLIO OPTIMISATION")
        print(f"{'=' * 70}")

        forecast_df = forecast_df.drop_duplicates(subset='ticker')
        forecast_df = forecast_df[forecast_df['ticker'].isin(tickers_raw)].set_index('ticker')
        forecast_df = forecast_df.dropna(subset=['forecast_return', 'test_rmse'])

        df, S, tickers = self.align_data(forecast_df)

        views = (df['forecast_return'] / 100).to_dict()
        rmses = df['test_rmse'].values

        mcaps = self.get_market_caps(tickers)
        pi = black_litterman.market_implied_prior_returns(
            mcaps,
            risk_aversion = self.risk_aversion,
            cov_matrix = S
        )

        omega = self.calculate_omega(S, rmses, tickers)

        bl = BlackLittermanModel(
            S,
            pi = pi,
            absolute_views = views,
            omega = omega,
            tau = self.tau
        )

        self.last_bl_model = bl

        ret_bl = bl.bl_returns()
        S_bl = bl.bl_cov()

        ef = EfficientFrontier(ret_bl, S_bl)
        ef.add_constraint(lambda w: w >= 0)  # No shorting

        if method == 'max_sharpe':
            ef.max_sharpe()
        elif method == 'min_volatility':
            ef.min_volatility()
        elif method == 'max_quadratic_utility':
            ef.max_quadratic_utility(risk_aversion=self.risk_aversion)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.last_ef = ef

        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)

        self.last_weights = weights
        self.last_performance = performance

        diagnostics = self.generate_diagnostics(df, S, rmses, omega, ret_bl)

        self.print_report(weights, performance, diagnostics, df)

        return {
            'weights': weights,
            'performance': performance,
            'bl_returns': ret_bl,
            'diagnostics': diagnostics
        }

    def optimise_markowitz(self, forecast_df, method='max_sharpe'):
        print(f"\n{'=' * 70}")
        print(f"MARKOWITZ OPTIMISATION ({method.upper()})")
        print(f"{'=' * 70}")

        # Align data
        df, S, tickers = self.align_data(forecast_df)

        # Use forecasts as expected returns
        mu = df['forecast_return'] / 100  # Convert to decimal

        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0)  # No shorting

        if method == 'max_sharpe':
            ef.max_sharpe()
        elif method == 'min_volatility':
            ef.min_volatility()
        else:
            raise ValueError(f"Unknown method: {method}")

        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)

        self.last_weights = weights
        self.last_performance = performance

        return {'weights': weights, 'performance': performance}

    def generate_diagnostics(self, df, S, rmses, omega, ret_bl):
        """ Generate diagnostic information about the optimisation. """
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

# get cov_matrix
filepath = '../../data/ESGU_LSTM_Ready.csv'
covariance_calculator = CovarianceCalculator()
cov_matrix = covariance_calculator.from_csv(filepath)

# get tickers
df = pd.read_csv(filepath)
tickers_raw = sorted(df["Ticker"].unique())

forecast_df = pd.read_csv('../../results/results.csv')

portfolio_optimiser = PortfolioOptimiser(cov_matrix)
portfolio_optimiser.optimise_black_litterman(tickers_raw, forecast_df)

portfolio_optimiser.optimise_markowitz(forecast_df)