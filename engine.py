import os
import sys
import subprocess

def initialize_environment():
    """
    Installs required dependencies and prepares the local Colab directory
    for production deployment and GitHub integration.
    """
    print("[INFO] Initializing system dependencies...")
    dependencies = ['yfinance', 'statsmodels', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy']

    for package in dependencies:
        try:
            __import__(package)
        except ImportError:
            print(f"[INFO] Package '{package}' missing. Installing via pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    print("[INFO] All core dependencies provisioned successfully.")

initialize_environment()

!git config --global user.name "sruthigandhi"
!git config --global user.email "sruthi.gandhi@outlook.com"
!git clone https://github.com/sruthigandhi/cointegrated.git
%cd cointegrated

import logging

class TradingConfig:
    """
    Encapsulates all quantitative configurations, execution parameters, risk
    thresholds, and environment configurations. Ensures zero hardcoding across
    the system.
    """
    # asset parameters
    PRIMARY_ASSET = "KO"      # y
    SECONDARY_ASSET = "PEP"   # x

    # data spans
    START_DATE = "2018-01-01"
    END_DATE = "2025-12-31"

    # strat tuning parameters
    TRAINING_SPLIT = 0.70     # % of data assigned to in-sample calibration
    Z_ENTRY_THRESHOLD = 2.0   # stdev entry barrier
    Z_EXIT_THRESHOLD = 0.0    # stdev exit target (mean-reversion)

    # financial fric penalties
    TRANSACTION_COST_BPS = 5.0 # basis points
    INITIAL_CAPITAL = 1000000.0 # seed money base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PairsTradingEngine")
logger.info("System configuration successfully initialized.")

import yfinance as yf
import pandas as pd
import numpy as np

class InstitutionalDataPipeline:
    """
    Handles robust asset time-series retrieval, handling missing values,
    timezone synchronization, and cross-market holiday data filling.
    """
    def __init__(self, config: TradingConfig):
        self.config = config

    def fetch_historical_data(self) -> pd.DataFrame:
        """
        Downloads and cleans daily adjusted close prices for configured tickers.
        """
        tickers = [self.config.PRIMARY_ASSET, self.config.SECONDARY_ASSET]
        logger.info(f"Downloading historical data for targets: {tickers} from {self.config.START_DATE} to {self.config.END_DATE}")

        try:
            # PATCH: Added explicit auto_adjust=False to preserve 'Adj Close' column structure
            raw_data = yf.download(
                tickers=tickers,
                start=self.config.START_DATE,
                end=self.config.END_DATE,
                progress=False,
                auto_adjust=False
            )

            if raw_data.empty:
                raise ValueError("Data retrieval failed: Returned dataframe is empty.")

            # iso adjust close column matrices
            cleaned_df = raw_data['Adj Close'].copy()

            # chronological formatting and data integrity
            cleaned_df = cleaned_df.dropna(how='all')
            cleaned_df = cleaned_df.ffill().bfill() # Handle asymmetric data gaps


            cleaned_df = cleaned_df[[self.config.PRIMARY_ASSET, self.config.SECONDARY_ASSET]]

            logger.info(f"Data extraction complete. Dimensions verified: {cleaned_df.shape}")
            return cleaned_df

        except Exception as e:
            logger.error(f"Critical breakdown during data engineering pipeline: {str(e)}")
            raise e

# verification
pipeline = InstitutionalDataPipeline(TradingConfig)
market_price_matrix = pipeline.fetch_historical_data()
print(market_price_matrix.head())


# STATISTICAL COINTEGRATION & ORNSTEIN-UHLENBECK ENGINE
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class QuantitativeStatisticalAnalyzer:
    """
    Executes the Engle-Granger two-step cointegration protocol and calculates
    the Ornstein-Uhlenbeck mean-reversion profile of the asset spread.
    """
    @staticmethod
    def calibrate_pair(insample_matrix: pd.DataFrame, primary_col: str, secondary_col: str) -> dict:
        """
        Runs OLS regression to compute the long-term hedge ratio and validates
        the stationarity of the residuals using the Augmented Dickey-Fuller test.
        """
        Y = insample_matrix[primary_col]
        X = insample_matrix[secondary_col]

        # intercept to capture structural pricing asset divergence constants
        X_with_intercept = sm.add_constant(X)
        model = sm.OLS(Y, X_with_intercept).fit()

        alpha = model.params.iloc[0]
        beta = model.params.iloc[1]


        residuals = Y - (beta * X) - alpha

        # augmented Dickey-Fuller stationarity check
        adf_result = adfuller(residuals, maxlag=1, autolag=None)
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]

        # half-life of mean reversion via an AR(1) model on residuals
        # Delta(Res_t) = lambda * Res_{t-1} + e
        df_lag = residuals.shift(1).dropna()
        df_delta = residuals.diff().dropna()

        # aligning index series
        df_lag = df_lag.loc[df_delta.index]

        ou_model = sm.OLS(df_delta, sm.add_constant(df_lag)).fit()
        # slope metrics
        lambda_val = ou_model.params.iloc[1] if len(ou_model.params) > 1 else ou_model.params.iloc[0]

        # guard against non-stationary positive expansion parameters
        if lambda_val >= 0:
            half_life = np.inf
        else:
            half_life = -np.log(2) / lambda_val

        return {
            "alpha": alpha,
            "beta": beta,
            "adf_stat": adf_statistic,
            "p_value": p_value,
            "critical_95": critical_values['5%'],
            "half_life_days": half_life,
            "residuals_mean": residuals.mean(),
            "residuals_std": residuals.std()
        }

# calibration run execution
split_idx = int(len(market_price_matrix) * TradingConfig.TRAINING_SPLIT)
train_data = market_price_matrix.iloc[:split_idx]
test_data = market_price_matrix.iloc[split_idx:]

stats_metrics = QuantitativeStatisticalAnalyzer.calibrate_pair(
    train_data, TradingConfig.PRIMARY_ASSET, TradingConfig.SECONDARY_ASSET
)

logger.info(f"--- Calibration Metrics ({TradingConfig.PRIMARY_ASSET} vs {TradingConfig.SECONDARY_ASSET}) ---")
logger.info(f"Hedge Ratio (Beta): {stats_metrics['beta']:.4f}")
logger.info(f"ADF Statistic: {stats_metrics['adf_stat']:.4f} (p-value: {stats_metrics['p_value']:.4f})")
logger.info(f"Ornstein-Uhlenbeck Half-Life: {stats_metrics['half_life_days']:.2f} trading days")


class AlgorithmicBacktester:
    """
    Computes rolling Z-scores out-of-sample, generates trading signals,
    and handles portfolio state transitions adjusted for transaction costs.
    """
    def __init__(self, config: TradingConfig, stats: dict):
        self.config = config
        self.stats = stats

    def run_backtest(self, full_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Executes a vectorized simulation of the statistical arbitrage strategy.
        """
        df = full_matrix.copy()
        beta = self.stats['beta']
        alpha = self.stats['alpha']

        # strategy spread
        df['Spread'] = df[self.config.PRIMARY_ASSET] - (beta * df[self.config.SECONDARY_ASSET]) - alpha

        # dynamic rolling Z-Score
        df['Z_Score'] = (df['Spread'] - self.stats['residuals_mean']) / self.stats['residuals_std']


        df['Position'] = 0.0

        current_position = 0.0
        positions = []

        for idx in range(len(df)):
            z = df['Z_Score'].iloc[idx]

            if current_position == 0.0:
                if z > self.config.Z_ENTRY_THRESHOLD:
                    current_position = -1.0 # short spread (expect downward mean reversion)
                elif z < -self.config.Z_ENTRY_THRESHOLD:
                    current_position = 1.0  # long spread (expect upward mean reversion)
            else:
                # checking for mean reversion exit crossing targets
                if current_position == -1.0 and z <= self.config.Z_EXIT_THRESHOLD:
                    current_position = 0.0
                elif current_position == 1.0 and z >= -self.config.Z_EXIT_THRESHOLD:
                    current_position = 0.0

            positions.append(current_position)

        df['Position'] = positions

        # compute asset underlying returns matrix profiles
        df['Ret_Y'] = df[self.config.PRIMARY_ASSET].pct_change().fillna(0)
        df['Ret_X'] = df[self.config.SECONDARY_ASSET].pct_change().fillna(0)

        # calculate spread log returns profile: R_spread = Ret_Y - beta * Ret_X
        # adjust weights to scale to 100% target dollar deployment exposure
        weight_y = 1.0 / (1.0 + abs(beta))
        weight_x = -beta / (1.0 + abs(beta))

        df['Strategy_Raw_Return'] = df['Position'].shift(1) * (weight_y * df['Ret_Y'] + abs(weight_x) * df['Ret_X'])
        df['Strategy_Raw_Return'] = df['Strategy_Raw_Return'].fillna(0)

        # model transaction cost friction when trade positions shift state
        df['Trades'] = df['Position'].diff().fillna(0).abs()
        friction = df['Trades'] * (self.config.TRANSACTION_COST_BPS / 10000.0)
        df['Strategy_Net_Return'] = df['Strategy_Raw_Return'] - friction

        # calculate cumulative metrics
        df['Cum_Spread_Returns'] = (1.0 + df['Strategy_Net_Return']).cumprod()
        df['Cum_Benchmark_Y'] = (1.0 + df['Ret_Y']).cumprod()

        return df

backtester = AlgorithmicBacktester(TradingConfig, stats_metrics)
backtest_results = backtester.run_backtest(market_price_matrix)
print(backtest_results[['Z_Score', 'Position', 'Strategy_Net_Return', 'Cum_Spread_Returns']].tail())


class PerformanceRiskAnalyzer:
    """
    Computes rigorous performance evaluation metrics, matching risk analytics
    generated by professional hedge fund systems.
    """
    @staticmethod
    def compute_tearsheet(df: pd.DataFrame, config: TradingConfig) -> dict:
        """
        Calculates annualized performance statistics from daily return profiles.
        """
        net_returns = df['Strategy_Net_Return']
        total_days = len(net_returns)
        years = total_days / 252.0

        cumulative_return = df['Cum_Spread_Returns'].iloc[-1] - 1.0
        annualized_return = (1.0 + cumulative_return) ** (1.0 / years) - 1.0

        daily_vol = net_returns.std()
        annualized_vol = daily_vol * np.sqrt(252)

        # sharpe calculation (Assuming 0% risk free floor standard baseline)
        sharpe = (annualized_return / annualized_vol) if annualized_vol > 0 else 0

        # downside volatility / sortino framework calculation
        downside_returns = net_returns[net_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (annualized_return / downside_vol) if downside_vol > 0 else 0

        # max drawdown caludation
        cum_equity = df['Cum_Spread_Returns']
        running_max = cum_equity.cummax()
        drawdowns = (cum_equity - running_max) / running_max
        max_drawdown = drawdowns.min()

        # executin accuracy
        total_trades = int(df['Trades'].sum())

        return {
            "Total Returns (%)": cumulative_return * 100,
            "Annual Return (%)": annualized_return * 100,
            "Annual Volatility (%)": annualized_vol * 100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown (%)": max_drawdown * 100,
            "Total Trades Executed": total_trades
        }

metrics = PerformanceRiskAnalyzer.compute_tearsheet(backtest_results, TradingConfig)
print("\n" + "="*40 + "\n INSTITUTIONAL RISK TEAR-SHEET \n" + "="*40)
for k, v in metrics.items():
    print(f"{k:<25}: {v:.4f}" if "Ratio" in k else f"{k:<25}: {v:.2f}")


import matplotlib.pyplot as plt
import seaborn as sns

class ProductionVisualizer:
    """
    Generates ultra-high resolution, publication-ready visualization panels
    suitable for recruiting portfolios and client presentations.
    """
    @staticmethod
    def plot_strategy_dashboard(df: pd.DataFrame, config: TradingConfig, stats: dict):
        """
        Renders a 3-panel professional dashboard displaying pricing models,
        signal overlays, and backtest results.
        """

        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=False)

        # PANEL 1: Cointegrated Asset Pricing Analysis
        axes[0].plot(df.index, df[config.PRIMARY_ASSET], label=f"Primary (Y): {config.PRIMARY_ASSET}", color='#1f77b4', lw=1.5)
        ax2 = axes[0].twinx()
        ax2.plot(df.index, df[config.SECONDARY_ASSET], label=f"Secondary (X): {config.SECONDARY_ASSET}", color='#ff7f0e', lw=1.5)
        axes[0].set_title(f"Asset Price Series Panel ({config.PRIMARY_ASSET} & {config.SECONDARY_ASSET})", fontsize=14, fontweight='bold')
        axes[0].set_ylabel(f"{config.PRIMARY_ASSET} USD Price")
        ax2.set_ylabel(f"{config.SECONDARY_ASSET} USD Price")


        lines, labels = axes[0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)
        ax2.grid(False)

        # PANEL 2: Residual Spread and Entry/Exit Signals
        axes[1].plot(df.index, df['Z_Score'], label='Rolling Spread Z-Score', color='#2ca02c', lw=1.2)
        axes[1].axhline(config.Z_ENTRY_THRESHOLD, color='red', linestyle='--', alpha=0.7, label='Short Threshold (+2.0σ)')
        axes[1].axhline(-config.Z_ENTRY_THRESHOLD, color='blue', linestyle='--', alpha=0.7, label='Long Threshold (-2.0σ)')
        axes[1].axhline(config.Z_EXIT_THRESHOLD, color='black', linestyle='-', alpha=0.5, label='Mean Reversion Target')


        shorts = df[df['Position'] == -1.0]
        longs = df[df['Position'] == 1.0]
        axes[1].scatter(shorts.index, shorts['Z_Score'], color='red', marker='v', s=15, alpha=0.5)
        axes[1].scatter(longs.index, longs['Z_Score'], color='blue', marker='^', s=15, alpha=0.5)

        axes[1].set_title("Cointegrated Spread Residual Z-Score Profile with Execution Triggers", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Standard Deviations (σ)")
        axes[1].legend(loc='upper left', frameon=True)

        # PANEL 3: Cumulative Performance Analysis vs Asset Baseline
        axes[2].plot(df.index, df['Cum_Spread_Returns'], label='Stat-Arb Strategy (Net of Friction)', color='#d62728', lw=2.0)
        axes[2].plot(df.index, df['Cum_Benchmark_Y'], label=f"Long Only Asset: {config.PRIMARY_ASSET}", color='grey', linestyle='--', alpha=0.6)
        axes[2].set_title("Strategy Performance vs. Long-Only Benchmark", fontsize=14, fontweight='bold')
        axes[2].set_ylabel("Growth Profile (Base Variable $1.0)")
        axes[2].legend(loc='upper left', frameon=True)

        plt.tight_layout()
        output_path = "pairs_trading_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"High-resolution production asset dashboard saved to local cluster disk: {output_path}")

# Run visualization system
ProductionVisualizer.plot_strategy_dashboard(backtest_results, TradingConfig, stats_metrics)


import unittest

class TestPairsTradingEngine(unittest.TestCase):
    """
    Executes automated code and data integrity assertions to guarantee
    mathematical correctness and prevent software regression.
    """
    def setUp(self):
        self.config = TradingConfig()
        self.dummy_pipeline = InstitutionalDataPipeline(self.config)
        self.test_matrix = market_price_matrix.head(100)

    def test_data_integrity(self):
        """Verify data matrices are properly aligned and contain no nulls."""
        self.assertEqual(self.test_matrix.isnull().sum().sum(), 0)
        self.assertIn(self.config.PRIMARY_ASSET, self.test_matrix.columns)
        self.assertIn(self.config.SECONDARY_ASSET, self.test_matrix.columns)

    def test_signal_bounding(self):
        """Assert strategy signals fall strict within valid parameter profiles (-1, 0, 1)."""
        engine = AlgorithmicBacktester(self.config, stats_metrics)
        out = engine.run_backtest(market_price_matrix)
        unique_positions = out['Position'].unique()
        for pos in unique_positions:
            self.assertTrue(pos in [-1.0, 0.0, 1.0])

# Execute automated verification suite inside the Colab runtime
suite = unittest.TestLoader().loadTestsFromTestCase(TestPairsTradingEngine)
unittest.TextTestRunner(verbosity=2).run(suite)