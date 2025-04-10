#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import time
import json
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class SmartOptimizer:
    """
    Fast and robust optimization framework for trading strategies.
    Uses intelligent sampling and parallelization to efficiently find optimal parameters.
    """
    def __init__(self, data_folder="./data"):
        """Initialize the optimizer with data folder path"""
        self.data_folder = data_folder
        self.products = ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK']
        self.price_data = {}  # Dictionary to store price data by product
        self.trade_data = {}  # Dictionary to store trade data by product
        
    def load_data(self):
        """Load market data from files with robust error handling"""
        print("Loading data...")
        price_files = []
        trade_files = []
        
        # Find data files in various locations
        day_indices = [-2, -1, 0]  # Historical days
        
        # Potential locations to search for data files
        locations = [
            self.data_folder,
            os.path.join(self.data_folder, "round-1-island-data-bottle"),
            "round-1-island-data-bottle",
            "."  # Current directory
        ]
        
        # Find all data files
        for day_idx in day_indices:
            price_found = False
            trade_found = False
            
            for loc in locations:
                # Check price files
                price_path = os.path.join(loc, f"prices_round_1_day_{day_idx}.csv")
                if os.path.exists(price_path) and not price_found:
                    price_files.append((day_idx, price_path))
                    price_found = True
                    print(f"Found price data for day {day_idx}: {price_path}")
                
                # Check trade files
                trade_path = os.path.join(loc, f"trades_round_1_day_{day_idx}.csv")
                if os.path.exists(trade_path) and not trade_found:
                    trade_files.append((day_idx, trade_path))
                    trade_found = True
                    print(f"Found trade data for day {day_idx}: {trade_path}")
        
        if not price_files:
            raise Exception("No price data files found. Please check data folder.")
        
        # Load data for each product separately
        for product in self.products:
            print(f"Processing {product} data...")
            product_price_data = []
            
            # Load price data for this product
            for day_idx, file_path in price_files:
                try:
                    df = pd.read_csv(file_path, delimiter=';')
                    # Filter only this product's data
                    product_df = df[df['product'] == product].copy()
                    if not product_df.empty:
                        product_df['day'] = day_idx  # Add day column
                        product_price_data.append(product_df)
                        print(f"  Loaded {len(product_df)} rows from {file_path}")
                except Exception as e:
                    print(f"  Error loading {file_path}: {str(e)}")
            
            # Combine all days for this product
            if product_price_data:
                self.price_data[product] = pd.concat(product_price_data, ignore_index=True)
                self.price_data[product] = self.price_data[product].sort_values(['day', 'timestamp']).reset_index(drop=True)
                
                # Preprocess data right away
                self._preprocess_product_data(product)
                
                print(f"  Total {product} data: {len(self.price_data[product])} rows")
            else:
                print(f"  Warning: No data found for {product}")
    
    def _preprocess_product_data(self, product):
        """Preprocess a single product's data"""
        if product not in self.price_data:
            print(f"No data for {product}")
            return
        
        # Make sure we're working with a copy
        df = self.price_data[product]
        
        # Calculate mid price if not already present
        if 'mid_price' not in df.columns:
            df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
        
        # Calculate order book features
        df['total_bid_volume'] = df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1, skipna=True)
        df['total_ask_volume'] = df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1, skipna=True)
        
        # Calculate book imbalance safely
        df['imbalance_ratio'] = 0.0  # Default value
        total_volume = df['total_bid_volume'] + df['total_ask_volume']
        mask = total_volume > 0
        if mask.any():
            df.loc[mask, 'imbalance_ratio'] = (
                (df.loc[mask, 'total_bid_volume'] - df.loc[mask, 'total_ask_volume']) / 
                total_volume[mask]
            )
        
        # Store processed data back
        self.price_data[product] = df
    
    @dataclass
    class TradingState:
        """Data class to track state during backtest simulation"""
        position: int = 0
        cash: float = 0
        trades: List[Dict] = None
        entry_price: float = None
        
        def __post_init__(self):
            if self.trades is None:
                self.trades = []
    
    def backtest_mean_reversion(self, product, params, day_index=0):
        """Backtest a mean reversion strategy with given parameters"""
        if product not in self.price_data:
            return {'total_pnl': -999, 'total_trades': 0}  # Error case
        
        # Extract parameters
        window_size = int(params['window_size'])
        entry_threshold = float(params['entry_threshold'])
        exit_threshold = float(params['exit_threshold'])
        position_limit = int(params['position_limit'])
        order_size = int(params['order_size'])
        
        # Get product data for specific day
        df = self.price_data[product]
        day_df = df[df['day'] == day_index].copy()
        
        if day_df.empty:
            return {'total_pnl': -999, 'total_trades': 0}  # Error case
        
        # Calculate moving average with proper handling
        day_df['sma'] = day_df['mid_price'].rolling(window=window_size, min_periods=1).mean()
        # Replace any NaN values with the first valid value
        day_df['sma'] = day_df['sma'].fillna(method='bfill').fillna(day_df['mid_price'].iloc[0])
        
        # Calculate deviation safely
        day_df['deviation'] = 0.0
        nonzero_sma = day_df['sma'] != 0
        if nonzero_sma.any():
            day_df.loc[nonzero_sma, 'deviation'] = (
                (day_df.loc[nonzero_sma, 'mid_price'] - day_df.loc[nonzero_sma, 'sma']) / 
                day_df.loc[nonzero_sma, 'sma']
            )
        
        # Initialize trading state
        state = self.TradingState()
        positions = []
        pnl = []
        
        # Simulate trading
        for i in range(len(day_df)):
            row = day_df.iloc[i]
            
            # Get current values
            mid_price = row['mid_price']
            deviation = row['deviation']
            
            # Determine action based on strategy logic
            action = 0  # 0: no action, 1: buy, -1: sell
            
            # Entry logic
            if state.position == 0:
                if deviation < -entry_threshold:
                    action = 1  # buy when price is below MA
                elif deviation > entry_threshold:
                    action = -1  # sell when price is above MA
            # Exit logic
            else:
                if abs(deviation) < exit_threshold:
                    action = -state.position  # exit position when deviation is small
            
            # Execute trade if there's an action
            quantity = 0
            if action > 0:  # Buy
                quantity = min(order_size, position_limit - state.position)
                if quantity > 0:
                    state.position += quantity
                    state.cash -= quantity * mid_price
                    state.entry_price = mid_price
                    state.trades.append({
                        'action': 'buy',
                        'price': mid_price,
                        'quantity': quantity,
                        'timestamp': row['timestamp']
                    })
            elif action < 0:  # Sell
                quantity = min(order_size, position_limit + state.position)
                if quantity > 0:
                    state.position -= quantity
                    state.cash += quantity * mid_price
                    state.entry_price = mid_price if state.position != 0 else None
                    state.trades.append({
                        'action': 'sell',
                        'price': mid_price,
                        'quantity': quantity,
                        'timestamp': row['timestamp']
                    })
            
            # Record position and PnL
            positions.append(state.position)
            current_portfolio_value = state.cash + state.position * mid_price
            pnl.append(current_portfolio_value)
        
        # Force liquidation at the end to calculate final PnL
        final_price = day_df['mid_price'].iloc[-1]
        final_portfolio_value = state.cash + state.position * final_price
        
        # Calculate performance metrics
        return {
            'total_pnl': final_portfolio_value,
            'total_trades': len(state.trades),
            'sharpe_ratio': self._calculate_sharpe_ratio(pnl),
            'max_drawdown': self._calculate_max_drawdown(pnl),
            'positions': positions,
            'pnl_history': pnl
        }
    
    def backtest_order_book_imbalance(self, product, params, day_index=0):
        """Backtest an order book imbalance strategy with given parameters"""
        if product not in self.price_data:
            return {'total_pnl': -999, 'total_trades': 0}  # Error case
        
        # Extract parameters
        imbalance_threshold = float(params['imbalance_threshold'])
        take_profit = float(params['take_profit'])
        stop_loss = float(params['stop_loss'])
        position_limit = int(params['position_limit'])
        order_size = int(params['order_size'])
        
        # Get product data for specific day
        df = self.price_data[product]
        day_df = df[df['day'] == day_index].copy()
        
        if day_df.empty:
            return {'total_pnl': -999, 'total_trades': 0}  # Error case
        
        # Initialize trading state
        state = self.TradingState()
        positions = []
        pnl = []
        
        # Simulate trading
        for i in range(1, len(day_df)):
            row = day_df.iloc[i]
            
            # Get current values
            mid_price = row['mid_price']
            imbalance = row['imbalance_ratio']
            
            # Determine action based on strategy logic
            action = 0  # 0: no action, 1: buy, -1: sell
            
            # Check for take profit or stop loss if in position
            if state.position != 0 and state.entry_price is not None:
                price_change_pct = 100 * (mid_price - state.entry_price) / state.entry_price
                
                if state.position > 0:  # Long position
                    if price_change_pct >= take_profit or price_change_pct <= -stop_loss:
                        action = -1  # Close position
                else:  # Short position
                    if price_change_pct <= -take_profit or price_change_pct >= stop_loss:
                        action = 1  # Close position
            
            # Entry logic based on imbalance
            if action == 0 and state.position == 0 and abs(imbalance) > imbalance_threshold:
                if imbalance > 0:
                    action = 1  # Buy when more bids than asks
                else:
                    action = -1  # Sell when more asks than bids
            
            # Execute trade if there's an action
            quantity = 0
            if action > 0:  # Buy
                quantity = min(order_size, position_limit - state.position)
                if quantity > 0:
                    state.position += quantity
                    state.cash -= quantity * mid_price
                    state.entry_price = mid_price
                    state.trades.append({
                        'action': 'buy',
                        'price': mid_price,
                        'quantity': quantity,
                        'timestamp': row['timestamp']
                    })
            elif action < 0:  # Sell
                quantity = min(order_size, position_limit + state.position)
                if quantity > 0:
                    state.position -= quantity
                    state.cash += quantity * mid_price
                    state.entry_price = mid_price if state.position != 0 else None
                    state.trades.append({
                        'action': 'sell',
                        'price': mid_price,
                        'quantity': quantity,
                        'timestamp': row['timestamp']
                    })
            
            # Record position and PnL
            positions.append(state.position)
            current_portfolio_value = state.cash + state.position * mid_price
            pnl.append(current_portfolio_value)
        
        # Force liquidation at the end to calculate final PnL
        final_price = day_df['mid_price'].iloc[-1]
        final_portfolio_value = state.cash + state.position * final_price
        
        # Calculate performance metrics
        return {
            'total_pnl': final_portfolio_value,
            'total_trades': len(state.trades),
            'sharpe_ratio': self._calculate_sharpe_ratio(pnl),
            'max_drawdown': self._calculate_max_drawdown(pnl),
            'positions': positions,
            'pnl_history': pnl
        }
    
    def _calculate_sharpe_ratio(self, pnl_history):
        """Calculate Sharpe ratio from PnL history"""
        if len(pnl_history) < 2:
            return 0.0
        
        # Calculate returns
        returns = pd.Series(pnl_history).pct_change().dropna().values
        
        # Check if we have valid data
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Calculate Sharpe ratio
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        return sharpe
    
    def _calculate_max_drawdown(self, pnl_history):
        """Calculate maximum drawdown from PnL history"""
        if len(pnl_history) < 2:
            return 0.0
        
        # Convert to pandas Series for easier calculation
        equity_curve = pd.Series(pnl_history)
        
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown
        drawdown = (running_max - equity_curve) / running_max
        
        # Return maximum drawdown, handling NaN values
        max_dd = drawdown.max()
        return max_dd if not pd.isna(max_dd) else 0.0
    
    def _evaluate_params(self, args):
        """Helper function for parallel processing"""
        product, strategy, params, day_index = args
        
        if strategy == 'mean_reversion':
            result = self.backtest_mean_reversion(product, params, day_index)
        elif strategy == 'order_book_imbalance':
            result = self.backtest_order_book_imbalance(product, params, day_index)
        else:
            return {'params': params, 'score': -999}
        
        # Calculate score
        # Balance PnL, trading activity, risk-adjusted returns
        pnl = result['total_pnl']
        trades = result['total_trades']
        sharpe = max(0, result.get('sharpe_ratio', 0))
        drawdown = min(1, result.get('max_drawdown', 0))
        
        # Penalize no-trade strategies
        if trades == 0:
            score = -1000
        else:
            # Reward high PnL, high Sharpe, low drawdown, and active trading
            score = (pnl * (1 + 0.3 * sharpe) * (1 - 0.5 * drawdown) * 
                    (1 + 0.1 * min(trades, 20) / 20))  # Cap trade bonus at 20 trades
        
        return {'params': params, 'score': score, 'result': result}
    
    def optimize_parallel(self, product, strategy, param_grid, day_index=0, n_samples=100, n_jobs=None):
        """
        Run optimization in parallel using random sampling
        
        Args:
            product: Product to optimize
            strategy: Strategy type ('mean_reversion' or 'order_book_imbalance')
            param_grid: Dictionary of parameter ranges
            day_index: Day to use for optimization
            n_samples: Number of parameter combinations to try
            n_jobs: Number of parallel jobs (defaults to number of CPU cores)
        
        Returns:
            Optimization results dictionary
        """
        start_time = time.time()
        print(f"\nOptimizing {product} with {strategy} strategy ({n_samples} samples)...")
        
        # Generate parameter combinations using random sampling
        param_combinations = []
        for _ in range(n_samples):
            params = {}
            for param_name, param_range in param_grid.items():
                if isinstance(param_range[0], int):
                    # Integer parameter
                    params[param_name] = random.randint(param_range[0], param_range[1])
                else:
                    # Float parameter
                    params[param_name] = random.uniform(param_range[0], param_range[1])
            param_combinations.append((product, strategy, params, day_index))
        
        # Use multiprocessing to run evaluations in parallel
        n_jobs = n_jobs or min(cpu_count(), 8)  # Use max 8 cores to avoid overwhelming the system
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap(self._evaluate_params, param_combinations),
                total=len(param_combinations),
                desc=f"Testing {product} parameters"
            ))
        
        # Find best parameters
        best_result = max(results, key=lambda x: x['score'])
        
        # Run detailed backtest with best parameters
        if strategy == 'mean_reversion':
            final_result = self.backtest_mean_reversion(product, best_result['params'], day_index)
        else:
            final_result = self.backtest_order_book_imbalance(product, best_result['params'], day_index)
        
        elapsed_time = time.time() - start_time
        
        # Display results
        print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters for {product} {strategy} strategy:")
        for param, value in best_result['params'].items():
            print(f"  {param}: {value}")
        print(f"Best PnL: {final_result['total_pnl']:.2f}")
        print(f"Number of trades: {final_result['total_trades']}")
        print(f"Sharpe ratio: {final_result.get('sharpe_ratio', 0):.2f}")
        print(f"Max drawdown: {final_result.get('max_drawdown', 0):.2%}")
        
        # Create visualization
        if final_result['total_trades'] > 0:
            self._visualize_result(product, strategy, best_result['params'], final_result)
        
        return {
            'product': product,
            'strategy': strategy,
            'best_params': best_result['params'],
            'best_pnl': final_result['total_pnl'],
            'trades': final_result['total_trades'],
            'sharpe_ratio': final_result.get('sharpe_ratio', 0),
            'max_drawdown': final_result.get('max_drawdown', 0),
            'elapsed_time': elapsed_time
        }
    
    def _visualize_result(self, product, strategy, params, result):
        """Create visualizations for the backtest result"""
        # Create output directory if it doesn't exist
        os.makedirs("optimization_results", exist_ok=True)
        
        # Extract PnL history and positions
        pnl_history = result.get('pnl_history', [])
        positions = result.get('positions', [])
        
        if len(pnl_history) < 2 or len(positions) < 2:
            return
        
        try:
            # Create plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Convert to pandas Series for better plotting
            pnl_series = pd.Series(pnl_history)
            pos_series = pd.Series(positions)
            
            # Plot PnL
            axes[0].plot(pnl_series)
            axes[0].set_title(f"{product} {strategy} - PnL History")
            axes[0].set_ylabel("PnL")
            axes[0].grid(True)
            
            # Add stats as text
            max_drawdown = result.get('max_drawdown', 0)
            max_drawdown_str = f"{max_drawdown:.2%}" if not pd.isna(max_drawdown) else "N/A"
            
            stats_text = (
                f"Final PnL: {result['total_pnl']:.2f}\n"
                f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}\n"
                f"Max Drawdown: {max_drawdown_str}\n"
                f"Trades: {result['total_trades']}"
            )
            axes[0].text(0.02, 0.95, stats_text, transform=axes[0].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
            
            # Plot positions
            axes[1].plot(pos_series)
            axes[1].set_title(f"{product} {strategy} - Position History")
            axes[1].set_ylabel("Position")
            axes[1].set_xlabel("Time Step")
            axes[1].grid(True)
            
            # Add parameters as title
            params_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()])
            plt.suptitle(f"{product} {strategy} Strategy\nParameters: {params_str}", fontsize=12)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"optimization_results/{product}_{strategy}_result.png")
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to create visualization for {product}: {str(e)}")
            plt.close('all')  # Close any open figures
    
    def optimize_all_products(self, day_index=0, n_samples=100, n_jobs=None):
        """
        Optimize all products with appropriate strategies
        
        Args:
            day_index: Day to optimize for
            n_samples: Number of parameter combinations to try per product
            n_jobs: Number of parallel jobs for each product
        
        Returns:
            Dictionary of optimization results by product
        """
        print("\nStarting optimization for all products...")
        overall_start_time = time.time()
        
        # Parameter grids for each strategy
        mean_reversion_grid = {
            'window_size': (5, 40),  # More responsive windows
            'entry_threshold': (0.1, 1.5),  # Wider range for entry opportunities
            'exit_threshold': (0.05, 0.8),  # Exit thresholds
            'position_limit': (20, 50),  # Position limits
            'order_size': (5, 25)  # Order sizes
        }
        
        order_book_grid = {
            'imbalance_threshold': (0.05, 0.3),  # Imbalance sensitivity
            'take_profit': (0.5, 4.0),  # Take profit levels (percent)
            'stop_loss': (0.5, 3.0),  # Stop loss levels (percent)
            'position_limit': (20, 50),  # Position limits
            'order_size': (5, 25)  # Order sizes
        }
        
        # Results container
        results = {}
        
        # Optimize each product with appropriate strategy
        results['RAINFOREST_RESIN'] = self.optimize_parallel(
            'RAINFOREST_RESIN', 'mean_reversion', mean_reversion_grid, day_index, n_samples, n_jobs
        )
        
        results['KELP'] = self.optimize_parallel(
            'KELP', 'mean_reversion', mean_reversion_grid, day_index, n_samples, n_jobs
        )
        
        results['SQUID_INK'] = self.optimize_parallel(
            'SQUID_INK', 'order_book_imbalance', order_book_grid, day_index, n_samples, n_jobs
        )
        
        # Generate summary report
        self._generate_summary(results)
        
        # Print total time
        elapsed_time = time.time() - overall_start_time
        print(f"\nTotal optimization time: {elapsed_time:.2f} seconds")
        print("All results saved to optimization_results/ directory")
        
        return results
    
    def _generate_summary(self, results):
        """Generate summary report and visualizations"""
        # Create output directory if it doesn't exist
        os.makedirs("optimization_results", exist_ok=True)
        
        # Create summary dataframe
        summary_data = []
        for product, result in results.items():
            # Skip if it's the shared parameters
            if product == 'shared':
                continue
                
            summary_data.append({
                'Product': product,
                'Strategy': result['strategy'],
                'PnL': result['best_pnl'],
                'Trades': result['trades'],
                'Sharpe': result.get('sharpe_ratio', 0),
                'MaxDrawdown': result.get('max_drawdown', 0),
                'Time': result.get('elapsed_time', 0),
                **result['best_params']
            })
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_df.to_csv("optimization_results/optimization_summary.csv", index=False)
        
        # Save JSON for trading algorithm
        trading_params = {}
        for product, result in results.items():
            if product != 'shared':
                trading_params[product] = {
                    'strategy': result['strategy'],
                    **result['best_params']
                }
        
        # Save to JSON
        with open("optimization_results/optimization_params.json", "w") as f:
            json.dump(trading_params, f, indent=2)
        
        # Create summary text file
        with open("optimization_results/optimization_summary.txt", "w") as f:
            f.write("OPTIMIZATION SUMMARY\n")
            f.write("===================\n\n")
            
            for product, result in results.items():
                if product == 'shared':
                    continue
                    
                strategy = result['strategy']
                best_params = result['best_params']
                
                f.write(f"{product} - {strategy.upper()}\n")
                f.write("-" * 50 + "\n")
                f.write(f"PnL: {result['best_pnl']:.2f}\n")
                f.write(f"Trades: {result['trades']}\n")
                f.write(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}\n")
                f.write(f"Optimization Time: {result.get('elapsed_time', 0):.2f} seconds\n")
                f.write("Parameters:\n")
                
                for param, value in best_params.items():
                    f.write(f"  {param}: {value}\n")
                
                f.write("\n")
        
        # Create comparative visualization
        try:
            plt.figure(figsize=(12, 10))
            
            # Plot PnL comparison
            plt.subplot(2, 2, 1)
            sns.barplot(x='Product', y='PnL', data=summary_df)
            plt.title('PnL by Product')
            plt.ylabel('PnL')
            plt.grid(True, alpha=0.3)
            
            # Plot trade count
            plt.subplot(2, 2, 2)
            sns.barplot(x='Product', y='Trades', data=summary_df)
            plt.title('Trade Count by Product')
            plt.grid(True, alpha=0.3)
            
            # Plot Sharpe ratios
            plt.subplot(2, 2, 3)
            sns.barplot(x='Product', y='Sharpe', data=summary_df)
            plt.title('Sharpe Ratio by Product')
            plt.grid(True, alpha=0.3)
            
            # Plot max drawdowns
            plt.subplot(2, 2, 4)
            sns.barplot(x='Product', y='MaxDrawdown', data=summary_df)
            plt.title('Max Drawdown by Product')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("optimization_results/product_comparison.png")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create comparison visualization: {str(e)}")
        
        print("\nSummary report generated in optimization_results directory")

def main():
    """Run the optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parallel optimization for trading strategies')
    parser.add_argument('--data-folder', type=str, default='./data', help='Path to data folder')
    parser.add_argument('--day', type=int, default=0, help='Day index to optimize for (0, -1, or -2)')
    parser.add_argument('--samples', type=int, default=100, help='Number of parameter samples to test per product')
    parser.add_argument('--jobs', type=int, default=None, help='Number of parallel jobs (default: auto)')
    parser.add_argument('--product', type=str, default=None, help='Specific product to optimize (optional)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("optimization_results", exist_ok=True)
    
    # Create and initialize optimizer
    optimizer = SmartOptimizer(data_folder=args.data_folder)
    optimizer.load_data()
    
    # Run optimization
    if args.product:
        # Optimize a specific product
        if args.product == 'RAINFOREST_RESIN' or args.product == 'KELP':
            grid = {
                'window_size': (5, 40),
                'entry_threshold': (0.1, 1.5),
                'exit_threshold': (0.05, 0.8),
                'position_limit': (20, 50),
                'order_size': (5, 25)
            }
            optimizer.optimize_parallel(args.product, 'mean_reversion', grid, args.day, args.samples, args.jobs)
        elif args.product == 'SQUID_INK':
            grid = {
                'imbalance_threshold': (0.05, 0.3),
                'take_profit': (0.5, 4.0),
                'stop_loss': (0.5, 3.0),
                'position_limit': (20, 50),
                'order_size': (5, 25)
            }
            optimizer.optimize_parallel(args.product, 'order_book_imbalance', grid, args.day, args.samples, args.jobs)
        else:
            print(f"Unknown product: {args.product}")
    else:
        # Optimize all products
        optimizer.optimize_all_products(args.day, args.samples, args.jobs)
    
    print("\nOptimization complete. You can now use the optimized parameters in your trading algorithm.")

if __name__ == "__main__":
    main()