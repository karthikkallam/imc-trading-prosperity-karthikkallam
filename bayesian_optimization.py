#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF

class BayesianOptimizer:
    """
    Optimization framework for trading strategies using Bayesian optimization.
    Efficiently finds optimal parameters using Gaussian Processes and Expected Improvement.
    """
    def __init__(self, data_folder="./data"):
        """
        Initialize the optimizer with data folder path
        """
        self.data_folder = data_folder
        self.products = ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK']
        self.price_data = None
        self.trade_data = None
        self.product_data = {}
        
    def load_data(self):
        """
        Load market data from files with efficient handling
        """
        print("Loading data...")
        price_files = []
        trade_files = []
        
        # Search for data files in various locations
        day_indices = [-2, -1, 0]  # Historical days
        
        for i in day_indices:
            # Look in different potential locations
            locations = [
                self.data_folder,
                os.path.join(self.data_folder, "round-1-island-data-bottle"),
                "round-1-island-data-bottle",
                ""  # Current directory
            ]
            
            price_found = False
            trade_found = False
            
            for loc in locations:
                # Check price files
                price_path = os.path.join(loc, f"prices_round_1_day_{i}.csv")
                if os.path.exists(price_path) and not price_found:
                    price_files.append((i, price_path))
                    price_found = True
                    
                # Check trade files
                trade_path = os.path.join(loc, f"trades_round_1_day_{i}.csv")
                if os.path.exists(trade_path) and not trade_found:
                    trade_files.append((i, trade_path))
                    trade_found = True
                
                # If both found, move to next day
                if price_found and trade_found:
                    break
        
        if not price_files:
            raise Exception("No price data files found. Please check data folder.")
        
        # Load price data efficiently
        all_price_data = []
        for day_idx, file in price_files:
            try:
                df = pd.read_csv(file, delimiter=';')
                df['day'] = day_idx  # Ensure day column exists
                all_price_data.append(df)
                print(f"Loaded price data from {file}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        if all_price_data:
            self.price_data = pd.concat(all_price_data, ignore_index=True)
            print(f"Total price data rows: {len(self.price_data)}")
        else:
            raise Exception("Failed to load any price data")
        
        # Load trade data
        all_trade_data = []
        for day_idx, file in trade_files:
            try:
                df = pd.read_csv(file, delimiter=';')
                df['day'] = day_idx  # Add day column if not present
                all_trade_data.append(df)
                print(f"Loaded trade data from {file}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        if all_trade_data:
            self.trade_data = pd.concat(all_trade_data, ignore_index=True)
            print(f"Total trade data rows: {len(self.trade_data)}")
        else:
            print("Warning: No trade data loaded")
        
        # Process data
        self._preprocess_data()
    
    def _preprocess_data(self):
        """
        Preprocess market data with advanced features
        """
        print("Preprocessing data...")
        if self.price_data is None:
            raise Exception("No data loaded. Call load_data() first.")
        
        # Calculate mid price if not present
        if 'mid_price' not in self.price_data.columns:
            self.price_data['mid_price'] = (self.price_data['bid_price_1'] + self.price_data['ask_price_1']) / 2
        
        # Process each product with vectorized operations for speed
        for product in self.products:
            # Filter data for this product
            product_df = self.price_data[self.price_data['product'] == product].copy()
            product_df = product_df.sort_values(['day', 'timestamp']).reset_index(drop=True)
            
            # Calculate order book features with vectorized operations
            product_df['total_bid_volume'] = product_df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1, skipna=True)
            product_df['total_ask_volume'] = product_df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1, skipna=True)
            
            # Calculate book imbalance efficiently
            total_volume = product_df['total_bid_volume'] + product_df['total_ask_volume']
            product_df['imbalance_ratio'] = np.where(
                total_volume > 0,
                (product_df['total_bid_volume'] - product_df['total_ask_volume']) / total_volume,
                0
            )
            
            # Store processed data
            self.product_data[product] = product_df
            print(f"Preprocessed {product} data: {len(product_df)} rows")
    
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
        """
        Backtest a mean reversion strategy with given parameters
        """
        if product not in self.product_data:
            raise Exception(f"No data for product {product}")
        
        # Extract parameters
        window_size = params['window_size']
        entry_threshold = params['entry_threshold']
        exit_threshold = params['exit_threshold']
        position_limit = params['position_limit']
        order_size = params.get('order_size', params.get('base_order_qty', 20))
        
        # Get product data for specific day
        product_df = self.product_data[product].copy()
        day_df = product_df[product_df['day'] == day_index].copy()
        if day_df.empty:
            return {'total_pnl': 0, 'total_trades': 0}
        
        day_df = day_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate moving average and deviation with better handling of initial values
        day_df['sma'] = day_df['mid_price'].rolling(window=window_size, min_periods=1).mean()
        # Fill initial NaN values with the first available moving average
        first_valid_ma = day_df['sma'].first_valid_index()
        if first_valid_ma is not None:
            day_df['sma'].fillna(day_df['sma'].iloc[first_valid_ma], inplace=True)
        day_df['deviation'] = (day_df['mid_price'] - day_df['sma']) / day_df['sma'].clip(lower=0.0001)
        
        # Initialize trading state
        state = self.TradingState()
        positions = []
        pnl = []
        
        # Start from beginning since we handled NaNs properly
        for i in range(0, len(day_df)):
            row = day_df.iloc[i]
            
            # Current price and deviation
            mid_price = row['mid_price']
            deviation = row['deviation']
            
            # Determine action
            action = 0  # 0: no action, 1: buy, -1: sell
            
            # Strategy logic
            if state.position == 0:
                if deviation < -entry_threshold:
                    action = 1  # buy
                elif deviation > entry_threshold:
                    action = -1  # sell
            else:
                # Exit position if deviation reverts enough
                if abs(deviation) < exit_threshold:
                    action = -state.position  # exit position
            
            # Execute trade if there's an action
            quantity = 0
            if action > 0:
                quantity = min(order_size, position_limit - state.position)
                if quantity > 0:
                    state.position += quantity
                    state.cash -= quantity * mid_price
                    state.entry_price = mid_price
                    state.trades.append({'action': 'buy', 'price': mid_price, 'quantity': quantity})
            elif action < 0:
                quantity = min(order_size, position_limit + state.position)
                if quantity > 0:
                    state.position -= quantity
                    state.cash += quantity * mid_price
                    state.entry_price = mid_price if state.position != 0 else None
                    state.trades.append({'action': 'sell', 'price': mid_price, 'quantity': quantity})
            
            # Record position and PnL after each step
            positions.append(state.position)
            current_portfolio_value = state.cash + state.position * mid_price
            pnl.append(current_portfolio_value)
        
        # Final PnL calculation
        final_price = day_df['mid_price'].iloc[-1] if not day_df.empty else 0
        portfolio_value = state.cash + state.position * final_price
        
        # Force liquidation at the end to properly evaluate strategy
        if state.position != 0:
            portfolio_value = state.cash + state.position * final_price
        
        return {
            'total_pnl': portfolio_value,
            'total_trades': len(state.trades),
            'sharpe_ratio': self._calculate_sharpe_ratio(pnl) if pnl else 0,
            'max_drawdown': self._calculate_max_drawdown(pnl) if pnl else 0,
            'positions': positions,
            'pnl_history': pnl
        }
    
    def backtest_order_book_imbalance(self, product, params, day_index=0):
        """
        Backtest an order book imbalance strategy with given parameters
        """
        if product not in self.product_data:
            raise Exception(f"No data for product {product}")
        
        # Extract parameters
        imbalance_threshold = params['imbalance_threshold']
        take_profit = params['take_profit']
        stop_loss = params['stop_loss']
        position_limit = params['position_limit']
        order_size = params.get('order_size', params.get('base_order_qty', 20))
        
        # Get product data for specific day
        product_df = self.product_data[product].copy()
        day_df = product_df[product_df['day'] == day_index].copy()
        if day_df.empty:
            return {'total_pnl': 0, 'total_trades': 0}
        
        day_df = day_df.sort_values('timestamp').reset_index(drop=True)
        
        # Initialize trading state
        state = self.TradingState()
        positions = []
        pnl = []
        
        # Simulate trading
        for i in range(1, len(day_df)):
            row = day_df.iloc[i]
            
            # Current price and imbalance
            mid_price = row['mid_price']
            imbalance = row['imbalance_ratio']
            
            # Determine action based on strategy
            action = 0  # 0: no action, 1: buy, -1: sell
            
            # Check for take profit or stop loss
            if state.position != 0 and state.entry_price is not None:
                price_change_pct = (mid_price - state.entry_price) / state.entry_price * 100
                if state.position > 0:  # Long position
                    if price_change_pct >= take_profit or price_change_pct <= -stop_loss:
                        action = -1  # Close position
                else:  # Short position
                    if price_change_pct <= -take_profit or price_change_pct >= stop_loss:
                        action = 1  # Close position
            
            # Entry logic based on imbalance
            if action == 0 and state.position == 0 and abs(imbalance) > imbalance_threshold:
                if imbalance > 0:
                    action = 1  # buy when more bids than asks
                else:
                    action = -1  # sell when more asks than bids
            
            # Execute trade if there's an action
            quantity = 0
            if action > 0:
                quantity = min(order_size, position_limit - state.position)
                if quantity > 0:
                    state.position += quantity
                    state.cash -= quantity * mid_price
                    state.entry_price = mid_price
                    state.trades.append({'action': 'buy', 'price': mid_price, 'quantity': quantity})
            elif action < 0:
                quantity = min(order_size, position_limit + state.position)
                if quantity > 0:
                    state.position -= quantity
                    state.cash += quantity * mid_price
                    state.entry_price = mid_price if state.position != 0 else None
                    state.trades.append({'action': 'sell', 'price': mid_price, 'quantity': quantity})
            
            # Record position and PnL
            positions.append(state.position)
            current_portfolio_value = state.cash + state.position * mid_price
            pnl.append(current_portfolio_value)
        
        # Final PnL calculation
        final_price = day_df['mid_price'].iloc[-1] if not day_df.empty else 0
        portfolio_value = state.cash + state.position * final_price
        
        # Force liquidation at the end
        if state.position != 0:
            portfolio_value = state.cash + state.position * final_price
        
        return {
            'total_pnl': portfolio_value,
            'total_trades': len(state.trades),
            'sharpe_ratio': self._calculate_sharpe_ratio(pnl) if pnl else 0,
            'max_drawdown': self._calculate_max_drawdown(pnl) if pnl else 0,
            'positions': positions,
            'pnl_history': pnl
        }
    
    def _calculate_sharpe_ratio(self, pnl_history):
        """Calculate Sharpe ratio from PnL history"""
        if len(pnl_history) < 2:
            return 0
        
        # Convert to returns
        returns = np.diff(pnl_history)
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
            
        # Calculate Sharpe ratio (using 0 as risk-free rate for simplicity)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        return sharpe
    
    def _calculate_max_drawdown(self, pnl_history):
        """Calculate maximum drawdown from PnL history"""
        if not pnl_history:
            return 0
            
        # Convert to numpy array
        pnl_array = np.array(pnl_history)
        
        # Check if all values are the same (no drawdown)
        if np.all(pnl_array == pnl_array[0]):
            return 0
            
        # If array is constant or decreasing, there's no drawdown
        if len(pnl_array) <= 1 or np.all(np.diff(pnl_array) <= 0):
            return 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(pnl_array)
        
        # Handle the case when running_max has zeros
        drawdown = np.zeros_like(pnl_array, dtype=float)
        mask = (running_max > 0) & (pnl_array <= running_max)  # Only calculate for valid points
        
        if np.any(mask):
            # Safely calculate drawdown only where we have valid values
            drawdown[mask] = (running_max[mask] - pnl_array[mask]) / running_max[mask]
            return np.nanmax(drawdown)  # Use nanmax to ignore NaNs
        else:
            return 0.0  # No valid drawdown points
    
    def _expected_improvement(self, X, X_sample, Y_sample, gpr, xi=0.01):
        """
        Computes the expected improvement acquisition function.
        
        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.
            
        Returns:
            Expected improvements at points X.
        """
        mu, sigma = gpr.predict(X, return_std=True)
        
        # Check if sigma is 0 (no uncertainty)
        with np.errstate(divide='ignore'):
            # Best observed value
            mu_sample_opt = np.max(Y_sample)
            
            # Compute improvement
            imp = mu - mu_sample_opt - xi
            
            # Compute Z-score
            Z = np.where(sigma > 0, imp / sigma, 0)
            
            # Expected improvement
            ei = np.where(sigma > 0, 
                        imp * norm.cdf(Z) + sigma * norm.pdf(Z),
                        0)
            
            return ei
    
    def bayesian_optimization(self, objective_func, param_ranges, n_iterations=30, n_initial_points=10, xi=0.01):
        """
        Perform Bayesian optimization to find the parameters that maximize an objective function.
        
        Args:
            objective_func: Function that takes parameters and returns a value to maximize
            param_ranges: Dictionary of parameter names and their (min, max) ranges
            n_iterations: Number of iterations for optimization
            n_initial_points: Number of random initial points to try
            xi: Exploitation-exploration trade-off parameter
            
        Returns:
            Dictionary of best parameters and their score
        """
        # Define parameter bounds and names
        bounds = np.array(list(param_ranges.values()))
        param_names = list(param_ranges.keys())
        
        # Dimensionality
        dim = len(bounds)
        
        # Initial sampling
        X_sample = np.zeros((n_initial_points, dim))
        for i in range(n_initial_points):
            for j in range(dim):
                if isinstance(bounds[j][0], int):
                    # Integer parameter
                    X_sample[i, j] = random.randint(bounds[j][0], bounds[j][1])
                else:
                    # Float parameter
                    X_sample[i, j] = random.uniform(bounds[j][0], bounds[j][1])
        
        # Evaluate the objective function at the initial points
        Y_sample = np.zeros((n_initial_points, 1))
        for i in range(n_initial_points):
            params = {param_names[j]: X_sample[i, j] for j in range(dim)}
            Y_sample[i] = objective_func(params)
            
        # Initialize progress bar
        pbar = tqdm(total=n_iterations, desc="Bayesian Optimization")
            
        # Define the kernel for the Gaussian Process
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(1e-5, 1e5))
        
        # Initialize Gaussian Process Regressor
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            alpha=1e-6  # Noise level
        )
        
        # Create grid for expected improvement calculation
        # For efficiency, we use a grid approach rather than randomized sampling
        X_grid = np.zeros((10000, dim))
        grid_points = max(10, int(10000 ** (1/dim)))  # Adjust grid size based on dimensionality
        
        # Current best solution
        best_idx = np.argmax(Y_sample)
        current_best_params = {param_names[j]: X_sample[best_idx, j] for j in range(dim)}
        current_best_score = Y_sample[best_idx][0]
        
        # Iterative optimization
        for i in range(n_iterations):
            # Update progress bar
            pbar.update(1)
            
            # Fit the Gaussian Process
            try:
                gpr.fit(X_sample, Y_sample)
            except Exception as e:
                print(f"Warning: GPR fitting failed with error: {str(e)}. Using current best.")
                continue
                
            # Generate random grid
            for j in range(dim):
                if isinstance(bounds[j][0], int):
                    # Integer parameter - uniform sampling
                    grid_values = np.random.randint(bounds[j][0], bounds[j][1] + 1, size=grid_points)
                else:
                    # Float parameter - uniform sampling within bounds
                    grid_values = np.random.uniform(bounds[j][0], bounds[j][1], size=grid_points)
                
                # Tile for other dimensions
                grid_values = np.tile(grid_values, grid_points ** (dim - 1 - j))
                grid_values = np.repeat(grid_values, grid_points ** j)
                X_grid[:, j] = grid_values[:10000]  # Ensure we don't exceed grid size
            
            # Compute expected improvement
            try:
                ei = self._expected_improvement(X_grid, X_sample, Y_sample, gpr, xi=xi)
            except Exception as e:
                print(f"Warning: EI computation failed with error: {str(e)}. Using random selection.")
                # Fallback to random selection
                next_sample_idx = np.random.randint(0, len(X_grid))
            else:
                # Find the point with maximum expected improvement
                next_sample_idx = np.argmax(ei)
            
            # Check if the selected point has already been sampled
            # We ensure some level of exploration by checking for duplicates
            next_sample = X_grid[next_sample_idx].reshape(1, -1)
            if np.any(np.all(X_sample == next_sample, axis=1)):
                # If already sampled, pick another point with EI in top 10%
                top_indices = np.argsort(ei)[-int(len(ei) * 0.1):]
                for alt_idx in top_indices:
                    alt_sample = X_grid[alt_idx].reshape(1, -1)
                    if not np.any(np.all(X_sample == alt_sample, axis=1)):
                        next_sample = alt_sample
                        break
            
            # Convert to parameters dictionary
            next_params = {}
            for j in range(dim):
                value = next_sample[0, j]
                # Round integer parameters
                if isinstance(bounds[j][0], int):
                    value = int(round(value))
                next_params[param_names[j]] = value
            
            # Evaluate the objective function at the new point
            next_score = objective_func(next_params)
            
            # Update samples
            X_sample = np.vstack((X_sample, next_sample))
            Y_sample = np.vstack((Y_sample, next_score))
            
            # Update best solution if needed
            if next_score > current_best_score:
                current_best_params = next_params
                current_best_score = next_score
                print(f"New best score: {current_best_score} with params: {current_best_params}")
        
        pbar.close()
        
        return {'params': current_best_params, 'score': current_best_score}
    
    def optimize_mean_reversion(self, product, day_index=0, n_iterations=30):
        """
        Optimize mean reversion strategy parameters using Bayesian optimization
        """
        print(f"\nOptimizing Mean Reversion strategy for {product} using Bayesian Optimization...")
        
        # Define parameter ranges
        param_ranges = {
            'window_size': (5, 40),  # More responsive window sizes
            'entry_threshold': (0.2, 1.5),  # Wider range to catch more opportunities
            'exit_threshold': (0.1, 0.8),  # Exit threshold should be smaller than entry
            'position_limit': (20, 50),
            'order_size': (5, 25)  # Base order quantity
        }
        
        # Define objective function
        def objective(params):
            result = self.backtest_mean_reversion(product, params, day_index)
            
            # Make sure we have valid metrics
            pnl = result['total_pnl']
            sharpe = max(0, result.get('sharpe_ratio', 0))
            drawdown = min(1, result.get('max_drawdown', 0))
            
            # If no trades were made, assign a very low score
            if result['total_trades'] == 0:
                return -1000  # Strong penalty for no trades
                
            # Combine PnL with Sharpe ratio for a more robust objective
            # We want high PnL with low drawdown and decent Sharpe ratio
            score = pnl * (1 + 0.2 * sharpe) * (1 - 0.5 * drawdown)
            
            return score
        
        # Run optimization
        start_time = time.time()
        result = self.bayesian_optimization(
            objective,
            param_ranges,
            n_iterations=n_iterations,
            n_initial_points=10
        )
        elapsed_time = time.time() - start_time
        
        # Get best parameters
        best_params = result['params']
        
        # Run detailed backtest with best parameters
        best_result = self.backtest_mean_reversion(product, best_params, day_index)
        
        print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters for {product} Mean Reversion strategy:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best PnL: {best_result['total_pnl']:.2f}")
        print(f"Number of trades: {best_result['total_trades']}")
        print(f"Sharpe ratio: {best_result['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {best_result['max_drawdown']:.2%}")
        
        # Visualize results
        self._visualize_backtest_result(best_result, product, "Mean_Reversion", best_params)
        
        return {
            'product': product,
            'strategy': 'mean_reversion',
            'best_params': best_params,
            'best_pnl': best_result['total_pnl'],
            'n_trades': best_result['total_trades'],
            'sharpe_ratio': best_result['sharpe_ratio'],
            'max_drawdown': best_result['max_drawdown'],
            'optimization_iterations': n_iterations,
            'elapsed_time': elapsed_time
        }
    
    def optimize_order_book_imbalance(self, product, day_index=0, n_iterations=30):
        """
        Optimize order book imbalance strategy parameters using Bayesian optimization
        """
        print(f"\nOptimizing Order Book Imbalance strategy for {product} using Bayesian Optimization...")
        
        # Define parameter ranges
        param_ranges = {
            'imbalance_threshold': (0.05, 0.25),  # Lower threshold to find more trading opportunities
            'take_profit': (0.8, 4.0),  # More realistic profit targets
            'stop_loss': (0.8, 2.5),  # More balanced risk management
            'position_limit': (20, 50),
            'order_size': (5, 25)  # Smaller orders can help find more trading opportunities
        }
        
        # Define objective function
        def objective(params):
            result = self.backtest_order_book_imbalance(product, params, day_index)
            
            # Make sure we have valid metrics
            pnl = result['total_pnl']
            sharpe = max(0, result.get('sharpe_ratio', 0))
            drawdown = min(1, result.get('max_drawdown', 0))
            
            # If no trades were made, assign a very low score
            if result['total_trades'] == 0:
                return -1000  # Strong penalty for no trades
                
            # Combine PnL with Sharpe ratio for a more robust objective
            # We want high PnL with low drawdown and decent Sharpe ratio
            score = pnl * (1 + 0.2 * sharpe) * (1 - 0.5 * drawdown)
            
            return score
        
        # Run optimization
        start_time = time.time()
        result = self.bayesian_optimization(
            objective,
            param_ranges,
            n_iterations=n_iterations,
            n_initial_points=10
        )
        elapsed_time = time.time() - start_time
        
        # Get best parameters
        best_params = result['params']
        
        # Run detailed backtest with best parameters
        best_result = self.backtest_order_book_imbalance(product, best_params, day_index)
        
        print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters for {product} Order Book Imbalance strategy:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best PnL: {best_result['total_pnl']:.2f}")
        print(f"Number of trades: {best_result['total_trades']}")
        print(f"Sharpe ratio: {best_result['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {best_result['max_drawdown']:.2%}")
        
        # Visualize results
        self._visualize_backtest_result(best_result, product, "Order_Book_Imbalance", best_params)
        
        return {
            'product': product,
            'strategy': 'order_book_imbalance',
            'best_params': best_params,
            'best_pnl': best_result['total_pnl'],
            'n_trades': best_result['total_trades'],
            'sharpe_ratio': best_result['sharpe_ratio'],
            'max_drawdown': best_result['max_drawdown'],
            'optimization_iterations': n_iterations,
            'elapsed_time': elapsed_time
        }
    
    def _visualize_backtest_result(self, result, product, strategy_name, params):
        """Create visualizations for the backtest result"""
        # Create output directory if it doesn't exist
        os.makedirs("optimization_results", exist_ok=True)
        
        # Extract PnL history and positions
        pnl_history = result.get('pnl_history', [])
        positions = result.get('positions', [])
        
        # Skip visualization if no meaningful data
        if not pnl_history or not positions or result['total_trades'] == 0:
            print(f"Skipping visualization for {product} - no meaningful trading activity")
            return
        
        try:
            # Create plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Plot PnL
            if pnl_history:
                axes[0].plot(pnl_history)
                axes[0].set_title(f"{product} {strategy_name} - PnL History")
                axes[0].set_ylabel("PnL")
                axes[0].grid(True)
                
                # Add stats as text
                max_drawdown = result.get('max_drawdown', 0)
                max_drawdown_str = f"{max_drawdown:.2%}" if not np.isnan(max_drawdown) else "N/A"
                
                stats_text = (
                    f"Final PnL: {result['total_pnl']:.2f}\n"
                    f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}\n"
                    f"Max Drawdown: {max_drawdown_str}\n"
                    f"Trades: {result['total_trades']}"
                )
                axes[0].text(0.02, 0.95, stats_text, transform=axes[0].transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
            
            # Plot positions
            if positions:
                axes[1].plot(positions)
                axes[1].set_title(f"{product} {strategy_name} - Position History")
                axes[1].set_ylabel("Position")
                axes[1].set_xlabel("Time Step")
                axes[1].grid(True)
            
            # Add parameters as title
            params_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()])
            plt.suptitle(f"{product} {strategy_name} Strategy\nParameters: {params_str}", fontsize=12)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"optimization_results/{product}_{strategy_name}_backtest.png")
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to create visualization for {product}: {str(e)}")
            plt.close('all')  # Close any open figures
    
    def optimize_all_products(self, day_index=0, n_iterations=30):
        """
        Optimize all products with their appropriate strategies
        """
        print("\nStarting optimization for all products...")
        optimization_results = {}
        
        # Track overall stats
        start_time = time.time()
        
        # Optimize RAINFOREST_RESIN with mean reversion
        rainforest_result = self.optimize_mean_reversion('RAINFOREST_RESIN', day_index, n_iterations)
        optimization_results['RAINFOREST_RESIN'] = rainforest_result
        
        # Optimize KELP with mean reversion
        kelp_result = self.optimize_mean_reversion('KELP', day_index, n_iterations)
        optimization_results['KELP'] = kelp_result
        
        # Optimize SQUID_INK with order book imbalance
        squid_result = self.optimize_order_book_imbalance('SQUID_INK', day_index, n_iterations)
        optimization_results['SQUID_INK'] = squid_result
        
        # Generate summary report
        self._generate_optimization_summary(optimization_results)
        
        # Print total time
        total_time = time.time() - start_time
        print(f"\nTotal optimization time: {total_time:.2f} seconds")
        
        return optimization_results
    
    def _generate_optimization_summary(self, optimization_results):
        """Generate comprehensive summary of optimization results"""
        # Create output directory if it doesn't exist
        os.makedirs("optimization_results", exist_ok=True)
        
        # Prepare summary data
        summary_data = []
        trading_params = {}
        
        # Process each product's results
        for product, result in optimization_results.items():
            best_params = result['best_params']
            strategy = result['strategy']
            
            # Add to summary data
            summary_data.append({
                'Product': product,
                'Strategy': strategy,
                'Best PnL': result['best_pnl'],
                'Sharpe Ratio': result.get('sharpe_ratio', 0),
                'Max Drawdown': result.get('max_drawdown', 0),
                'Trades': result.get('n_trades', 0),
                'Optimization Time': result.get('elapsed_time', 0),
                **best_params
            })
            
            # Add to trading parameters
            trading_params[product] = {
                'strategy': strategy,
                **best_params
            }
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_df.to_csv("optimization_results/bayesian_optimization_summary.csv", index=False)
        
        # Generate summary text file
        with open("optimization_results/bayesian_optimization_summary.txt", "w") as f:
            f.write("BAYESIAN OPTIMIZATION SUMMARY\n")
            f.write("===========================\n\n")
            
            for product, result in optimization_results.items():
                best_params = result['best_params']
                best_pnl = result['best_pnl']
                strategy = result['strategy']
                
                f.write(f"{product} - {strategy.upper()}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Best PnL: {best_pnl:.2f}\n")
                f.write(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}\n")
                f.write(f"Number of Trades: {result.get('n_trades', 0)}\n")
                f.write(f"Optimization Time: {result.get('elapsed_time', 0):.2f} seconds\n")
                f.write("Parameters:\n")
                
                for param, value in best_params.items():
                    f.write(f"  {param}: {value}\n")
                
                f.write("\n")
        
        # Save optimization results as JSON for the trading algorithm to use
        with open("optimization_results/optimization_summary.json", "w") as f:
            json.dump(trading_params, f, indent=2)
            
        print("\nOptimization summary saved to optimization_results/bayesian_optimization_summary.*")
        print("Trading parameters saved to optimization_results/optimization_summary.json")
        
        # Create comparative visualization
        self._create_comparative_visualization(summary_df)
    
    def _create_comparative_visualization(self, summary_df):
        """Create visualization comparing strategies across products"""
        plt.figure(figsize=(12, 8))
        
        # Bar chart of PnL by product/strategy
        plt.subplot(2, 1, 1)
        sns.barplot(x='Product', y='Best PnL', hue='Strategy', data=summary_df)
        plt.title('Performance Comparison by Product')
        plt.ylabel('PnL')
        plt.grid(True, alpha=0.3)
        
        # Risk metrics by product
        plt.subplot(2, 2, 3)
        sns.barplot(x='Product', y='Sharpe Ratio', data=summary_df)
        plt.title('Sharpe Ratio by Product')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        sns.barplot(x='Product', y='Max Drawdown', data=summary_df)
        plt.title('Max Drawdown by Product')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("optimization_results/bayesian_strategy_comparison.png")
        plt.close()

def main():
    """Main function to run the optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Bayesian optimization for trading strategies')
    parser.add_argument('--data-folder', type=str, default='./data', help='Path to data folder')
    parser.add_argument('--day', type=int, default=0, help='Day index to optimize for (default: 0)')
    parser.add_argument('--iterations', type=int, default=30, help='Number of optimization iterations (default: 30)')
    parser.add_argument('--product', type=str, default=None, help='Specific product to optimize (optional)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("optimization_results", exist_ok=True)
    
    # Create and setup optimizer
    optimizer = BayesianOptimizer(data_folder=args.data_folder)
    optimizer.load_data()
    
    start_time = time.time()
    
    # Run optimization for specific product or all products
    if args.product:
        if args.product == 'RAINFOREST_RESIN' or args.product == 'KELP':
            result = optimizer.optimize_mean_reversion(args.product, args.day, args.iterations)
        elif args.product == 'SQUID_INK':
            result = optimizer.optimize_order_book_imbalance(args.product, args.day, args.iterations)
        else:
            print(f"Unknown product: {args.product}")
            return
    else:
        # Optimize all products
        optimizer.optimize_all_products(args.day, args.iterations)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_time:.2f} seconds")
    print("\nOptimization results are available in the optimization_results directory")
    print("You can now use the optimized parameters in your trading algorithm")

if __name__ == "__main__":
    main()