#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import itertools
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

class StrategyOptimizer:
    def __init__(self, data_folder="./data"):
        """
        Initialize the optimizer with data folder path
        """
        self.data_folder = data_folder
        self.products = ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK']
        self.price_data = None
        self.trade_data = None
    
    def load_data(self):
        """
        Load price and trade data for all days
        """
        # Price files - Try from data folder first, then from round-1-island-data-bottle
        price_files = []
        trade_files = []
        
        # Check if files exist in data folder
        data_folder_exists = os.path.exists(self.data_folder)
        if not data_folder_exists:
            print(f"Warning: {self.data_folder} directory not found. Creating it...")
            os.makedirs(self.data_folder, exist_ok=True)
            
        # Try to find files for days -2, -1, 0 based on the actual data file naming
        day_indices = [-2, -1, 0]  # The actual day indices used in file names
        
        for i in day_indices:
            # Check potential file locations in priority order
            price_file_paths = [
                os.path.join(self.data_folder, "round-1-island-data-bottle", f"prices_round_1_day_{i}.csv"),
                os.path.join("round-1-island-data-bottle", f"prices_round_1_day_{i}.csv"),
                os.path.join(self.data_folder, f"prices_round_1_day_{i}.csv"),
                f"prices_round_1_day_{i}.csv"  # Current directory
            ]
            
            trade_file_paths = [
                os.path.join(self.data_folder, "round-1-island-data-bottle", f"trades_round_1_day_{i}.csv"),
                os.path.join("round-1-island-data-bottle", f"trades_round_1_day_{i}.csv"),
                os.path.join(self.data_folder, f"trades_round_1_day_{i}.csv"),
                f"trades_round_1_day_{i}.csv"  # Current directory
            ]
            
            # Find first existing price file
            price_file = None
            for path in price_file_paths:
                if os.path.exists(path):
                    price_file = path
                    break
            
            if price_file:
                price_files.append(price_file)
            else:
                print(f"Warning: Could not find prices_round_1_day_{i}.csv in any location")
            
            # Find first existing trade file
            trade_file = None
            for path in trade_file_paths:
                if os.path.exists(path):
                    trade_file = path
                    break
            
            if trade_file:
                trade_files.append(trade_file)
            else:
                print(f"Warning: Could not find trades_round_1_day_{i}.csv in any location")
        
        # Load all price data
        all_price_data = []
        for i, file in enumerate(price_files):
            try:
                df = pd.read_csv(file, delimiter=';')
                df['day_index'] = i
                all_price_data.append(df)
                print(f"Loaded price data from {file}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        if all_price_data:
            self.price_data = pd.concat(all_price_data, ignore_index=True)
            print(f"Total price data rows: {len(self.price_data)}")
        else:
            raise Exception("Failed to load any price data")
        
        # Load all trade data
        all_trade_data = []
        for i, file in enumerate(trade_files):
            try:
                df = pd.read_csv(file, delimiter=';')
                df['day_index'] = i
                all_trade_data.append(df)
                print(f"Loaded trade data from {file}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        if all_trade_data:
            self.trade_data = pd.concat(all_trade_data, ignore_index=True)
            print(f"Total trade data rows: {len(self.trade_data)}")
        else:
            print("Warning: No trade data loaded")
    
    def preprocess_data(self):
        """
        Preprocess the data for optimization
        """
        if self.price_data is None:
            raise Exception("No data loaded. Call load_data() first.")
        
        # Calculate mid price if not present
        if 'mid_price' not in self.price_data.columns:
            self.price_data['mid_price'] = (self.price_data['bid_price_1'] + self.price_data['ask_price_1']) / 2
        
        # Create separate DataFrames for each product
        self.product_data = {}
        for product in self.products:
            product_df = self.price_data[self.price_data['product'] == product].copy()
            product_df = product_df.sort_values(['day_index', 'timestamp'])
            
            # Calculate order book imbalance
            product_df['total_bid_volume'] = (
                product_df['bid_volume_1'].fillna(0) + 
                product_df['bid_volume_2'].fillna(0) + 
                product_df['bid_volume_3'].fillna(0)
            )
            
            product_df['total_ask_volume'] = (
                product_df['ask_volume_1'].fillna(0) + 
                product_df['ask_volume_2'].fillna(0) + 
                product_df['ask_volume_3'].fillna(0)
            )
            
            product_df['imbalance_ratio'] = (
                (product_df['total_bid_volume'] - product_df['total_ask_volume']) / 
                (product_df['total_bid_volume'] + product_df['total_ask_volume'])
            ).fillna(0)  # Handle division by zero
            
            # Store processed data
            self.product_data[product] = product_df
            print(f"Preprocessed {product} data: {len(product_df)} rows")
    
    def backtest_mean_reversion(self, product, params, visualize=False):
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
        order_size = params['order_size']
        
        # Get product data
        product_df = self.product_data[product].copy()
        
        # Prepare results by day
        results_by_day = {}
        
        # Process each day separately
        for day in product_df['day_index'].unique():
            day_df = product_df[product_df['day_index'] == day].copy()
            day_df = day_df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate moving average
            day_df['moving_avg'] = day_df['mid_price'].rolling(window=window_size).mean()
            day_df['deviation'] = day_df['mid_price'] - day_df['moving_avg']
            
            # Initialize trading state
            position = 0
            cash = 0
            trades = []
            positions = []
            pnl = []
            timestamps = []
            
            # Start trading from when we have enough data for MA
            for i in range(window_size, len(day_df)):
                row = day_df.iloc[i]
                
                # Current price and deviation
                mid_price = row['mid_price']
                deviation = row['deviation']
                timestamp = row['timestamp']
                
                # Determine buy/sell actions
                action = 0  # 0: no action, 1: buy, -1: sell
                
                # If no position, check if we should enter
                if position == 0:
                    if deviation < -entry_threshold:
                        action = 1  # buy
                    elif deviation > entry_threshold:
                        action = -1  # sell
                # If we have a position, check if we should exit
                else:
                    if abs(deviation) < exit_threshold:
                        action = -position  # exit position
                
                # Execute trade if there's an action
                quantity = 0
                if action > 0:
                    # Buy, limited by position limit
                    quantity = min(order_size, position_limit - position)
                    if quantity > 0:
                        position += quantity
                        cash -= quantity * mid_price
                        trades.append({
                            'timestamp': timestamp,
                            'action': 'buy',
                            'price': mid_price,
                            'quantity': quantity
                        })
                elif action < 0:
                    # Sell, limited by position limit
                    quantity = min(order_size, position_limit + position)
                    if quantity > 0:
                        position -= quantity
                        cash += quantity * mid_price
                        trades.append({
                            'timestamp': timestamp,
                            'action': 'sell',
                            'price': mid_price,
                            'quantity': quantity
                        })
                
                # Record position and PnL
                positions.append(position)
                portfolio_value = cash + position * mid_price
                pnl.append(portfolio_value)
                timestamps.append(timestamp)
            
            # Calculate performance metrics
            total_trades = len(trades)
            final_pnl = pnl[-1] if pnl else 0
            max_position = max(abs(p) for p in positions) if positions else 0
            
            # Store results for this day
            results_by_day[day] = {
                'total_trades': total_trades,
                'final_pnl': final_pnl,
                'max_position': max_position,
                'positions': positions,
                'pnl': pnl,
                'timestamps': timestamps
            }
        
        # Calculate overall performance
        total_pnl = sum(results_by_day[day]['final_pnl'] for day in results_by_day)
        total_trades = sum(results_by_day[day]['total_trades'] for day in results_by_day)
        
        # Create visualizations if requested
        if visualize:
            self._visualize_strategy_results(product, 'Mean Reversion', params, results_by_day)
        
        return {
            'strategy': 'mean_reversion',
            'params': params,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'results_by_day': results_by_day
        }
    
    def backtest_order_book_imbalance(self, product, params, visualize=False):
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
        order_size = params['order_size']
        
        # Get product data
        product_df = self.product_data[product].copy()
        
        # Prepare results by day
        results_by_day = {}
        
        # Process each day separately
        for day in product_df['day_index'].unique():
            day_df = product_df[product_df['day_index'] == day].copy()
            day_df = day_df.sort_values('timestamp').reset_index(drop=True)
            
            # Initialize trading state
            position = 0
            cash = 0
            trades = []
            positions = []
            pnl = []
            timestamps = []
            entry_price = None
            
            # Start trading
            for i in range(1, len(day_df)):
                row = day_df.iloc[i]
                prev_row = day_df.iloc[i-1]
                
                # Current price and imbalance
                mid_price = row['mid_price']
                imbalance = row['imbalance_ratio']
                timestamp = row['timestamp']
                
                # Determine buy/sell actions
                action = 0  # 0: no action, 1: buy, -1: sell
                
                # If we have a position, check if we should exit based on profit/loss
                if position != 0 and entry_price is not None:
                    price_change = mid_price - entry_price
                    directed_price_change = price_change if position > 0 else -price_change
                    
                    if directed_price_change >= take_profit or directed_price_change <= -stop_loss:
                        action = -position  # exit position
                
                # If no position, check if we should enter based on imbalance
                elif position == 0 and abs(imbalance) > imbalance_threshold:
                    if imbalance > 0:
                        action = 1  # buy
                    else:
                        action = -1  # sell
                
                # Execute trade if there's an action
                quantity = 0
                if action > 0:
                    # Buy, limited by position limit
                    quantity = min(order_size, position_limit - position)
                    if quantity > 0:
                        position += quantity
                        entry_price = mid_price
                        cash -= quantity * mid_price
                        trades.append({
                            'timestamp': timestamp,
                            'action': 'buy',
                            'price': mid_price,
                            'quantity': quantity
                        })
                elif action < 0:
                    # Sell, limited by position limit
                    quantity = min(order_size, position_limit + position)
                    if quantity > 0:
                        position -= quantity
                        entry_price = mid_price if position != 0 else None
                        cash += quantity * mid_price
                        trades.append({
                            'timestamp': timestamp,
                            'action': 'sell',
                            'price': mid_price,
                            'quantity': quantity
                        })
                
                # Record position and PnL
                positions.append(position)
                portfolio_value = cash + position * mid_price
                pnl.append(portfolio_value)
                timestamps.append(timestamp)
            
            # Calculate performance metrics
            total_trades = len(trades)
            final_pnl = pnl[-1] if pnl else 0
            max_position = max(abs(p) for p in positions) if positions else 0
            
            # Store results for this day
            results_by_day[day] = {
                'total_trades': total_trades,
                'final_pnl': final_pnl,
                'max_position': max_position,
                'positions': positions,
                'pnl': pnl,
                'timestamps': timestamps
            }
        
        # Calculate overall performance
        total_pnl = sum(results_by_day[day]['final_pnl'] for day in results_by_day)
        total_trades = sum(results_by_day[day]['total_trades'] for day in results_by_day)
        
        # Create visualizations if requested
        if visualize:
            self._visualize_strategy_results(product, 'Order Book Imbalance', params, results_by_day)
        
        return {
            'strategy': 'order_book_imbalance',
            'params': params,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'results_by_day': results_by_day
        }
    
    def _visualize_strategy_results(self, product, strategy_name, params, results_by_day):
        """
        Create visualizations for strategy backtest results
        """
        # Create a figure with subplots
        fig, axes = plt.subplots(len(results_by_day), 2, figsize=(15, 5 * len(results_by_day)))
        
        # Set the title for the entire figure
        fig.suptitle(f"{strategy_name} Strategy Backtest Results for {product}", fontsize=16)
        
        # If only one day, make axes a 2D array
        if len(results_by_day) == 1:
            axes = np.array([axes])
        
        # Plot results for each day
        for i, day in enumerate(sorted(results_by_day.keys())):
            day_results = results_by_day[day]
            
            # Plot PnL over time
            axes[i, 0].plot(day_results['timestamps'], day_results['pnl'])
            axes[i, 0].set_title(f"Day {day} - PnL over Time")
            axes[i, 0].set_xlabel("Timestamp")
            axes[i, 0].set_ylabel("PnL")
            axes[i, 0].grid(True)
            
            # Plot position over time
            axes[i, 1].plot(day_results['timestamps'], day_results['positions'])
            axes[i, 1].set_title(f"Day {day} - Position over Time")
            axes[i, 1].set_xlabel("Timestamp")
            axes[i, 1].set_ylabel("Position")
            axes[i, 1].grid(True)
            
            # Add final PnL and trade count annotation
            axes[i, 0].annotate(f"Final PnL: {day_results['final_pnl']:.2f}\nTrades: {day_results['total_trades']}", 
                             xy=(0.05, 0.95), xycoords='axes fraction',
                             ha='left', va='top',
                             bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        
        # Add parameters as text at the bottom
        param_text = ", ".join([f"{k}: {v}" for k, v in params.items()])
        fig.text(0.5, 0.01, f"Parameters: {param_text}", ha='center', fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Create output directory if it doesn't exist
        os.makedirs("optimization_results", exist_ok=True)
        
        # Save the figure
        plt.savefig(f"optimization_results/{product}_{strategy_name.replace(' ', '_')}_results.png")
        plt.close(fig)
    
    def grid_search(self, product, strategy, param_grid, visualize_best=True):
        """
        Perform grid search optimization for a given product and strategy
        """
        if product not in self.product_data:
            raise Exception(f"No data for product {product}")
        
        # Generate all parameter combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Initialize best result
        best_result = None
        best_pnl = float('-inf')
        all_results = []
        
        # Progress bar
        pbar = tqdm(total=len(param_combinations), desc=f"Optimizing {product} with {strategy}")
        
        # Test each parameter combination
        for combo in param_combinations:
            params = dict(zip(param_keys, combo))
            
            # Run backtest with the current parameters
            if strategy == 'mean_reversion':
                result = self.backtest_mean_reversion(product, params)
            elif strategy == 'order_book_imbalance':
                result = self.backtest_order_book_imbalance(product, params)
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            # Track results
            all_results.append({
                'params': params,
                'total_pnl': result['total_pnl'],
                'total_trades': result['total_trades']
            })
            
            # Update best result if this is better
            if result['total_pnl'] > best_pnl:
                best_pnl = result['total_pnl']
                best_result = result
            
            pbar.update(1)
        
        pbar.close()
        
        # Visualize the best result if requested
        if visualize_best and best_result:
            if strategy == 'mean_reversion':
                self.backtest_mean_reversion(product, best_result['params'], visualize=True)
            elif strategy == 'order_book_imbalance':
                self.backtest_order_book_imbalance(product, best_result['params'], visualize=True)
        
        # Visualize the parameter space
        self._visualize_param_space(product, strategy, all_results)
        
        return {
            'product': product,
            'strategy': strategy,
            'best_params': best_result['params'] if best_result else None,
            'best_pnl': best_pnl,
            'all_results': all_results
        }
    
    def _visualize_param_space(self, product, strategy, results):
        """
        Visualize parameter optimization results
        """
        # Convert results to DataFrame for easier visualization
        results_df = pd.DataFrame([
            {**r['params'], 'total_pnl': r['total_pnl'], 'total_trades': r['total_trades']}
            for r in results
        ])
        
        # Create output directory if it doesn't exist
        os.makedirs("optimization_results", exist_ok=True)
        
        # Get parameter names
        param_names = [col for col in results_df.columns if col not in ['total_pnl', 'total_trades']]
        
        # Create heatmaps for each pair of parameters
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names[i+1:], i+1):
                if len(results_df[param1].unique()) > 1 and len(results_df[param2].unique()) > 1:
                    plt.figure(figsize=(10, 8))
                    
                    # Create pivot table for heatmap
                    pivot = results_df.pivot_table(
                        values='total_pnl', 
                        index=param1, 
                        columns=param2, 
                        aggfunc='mean'
                    )
                    
                    # Create heatmap
                    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")
                    plt.title(f"{product} - {strategy}: Impact of {param1} vs {param2} on PnL")
                    
                    # Save the figure
                    plt.tight_layout()
                    plt.savefig(f"optimization_results/{product}_{strategy}_{param1}_vs_{param2}.png")
                    plt.close()
        
        # Create parameter importance chart
        plt.figure(figsize=(12, 8))
        
        # For each parameter, calculate its impact on PnL
        param_impacts = {}
        for param in param_names:
            if len(results_df[param].unique()) > 1:
                # Group by parameter value and calculate mean PnL
                param_pnl = results_df.groupby(param)['total_pnl'].mean()
                
                # Calculate range of PnL values
                param_impact = param_pnl.max() - param_pnl.min()
                param_impacts[param] = param_impact
                
                # Create line plot
                plt.figure(figsize=(10, 6))
                sns.lineplot(x=param, y='total_pnl', data=results_df, marker='o')
                plt.title(f"{product} - {strategy}: Impact of {param} on PnL")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"optimization_results/{product}_{strategy}_{param}_impact.png")
                plt.close()
        
        # Create parameter importance bar chart
        plt.figure(figsize=(10, 6))
        params = list(param_impacts.keys())
        impacts = list(param_impacts.values())
        
        # Sort by impact
        sorted_indices = np.argsort(impacts)
        sorted_params = [params[i] for i in sorted_indices]
        sorted_impacts = [impacts[i] for i in sorted_indices]
        
        plt.barh(sorted_params, sorted_impacts)
        plt.xlabel('Impact on PnL (Range)')
        plt.title(f"{product} - {strategy}: Parameter Importance")
        plt.tight_layout()
        plt.savefig(f"optimization_results/{product}_{strategy}_parameter_importance.png")
        plt.close()
        
        # Save all results to CSV
        results_df.to_csv(f"optimization_results/{product}_{strategy}_all_results.csv", index=False)
        
        # Find top 10 parameter combinations
        top_results = results_df.sort_values('total_pnl', ascending=False).head(10)
        top_results.to_csv(f"optimization_results/{product}_{strategy}_top_results.csv", index=False)
        
        # Print best parameters
        best_params = results_df.loc[results_df['total_pnl'].idxmax()]
        print(f"\nBest parameters for {product} - {strategy}:")
        for param in param_names:
            print(f"  {param}: {best_params[param]}")
        print(f"  Total PnL: {best_params['total_pnl']}")
        print(f"  Total Trades: {best_params['total_trades']}")
    
    def optimize_all_products(self):
        """
        Run optimization for all products with appropriate strategies
        """
        if not hasattr(self, 'product_data'):
            self.preprocess_data()
        
        optimization_results = {}
        
        # Define shared parameters grid
        shared_params_grid = {
            'take_profit_threshold': [0.3, 0.4, 0.5],
            'max_history_length': [60, 90, 120]
        }
        
        # Define parameter grids for RAINFOREST_RESIN mean reversion strategy
        rainforest_resin_grid = {
            'window_size': [20, 30, 40],
            'entry_threshold': [0.8, 1.0, 1.5],
            'exit_threshold': [0.5, 0.7, 0.9],
            'position_limit': [30, 40, 50],
            'order_size': [15, 20, 25],
            'fair_value_anchor': [9990.0, 10000.0, 10010.0],
            'anchor_blend_alpha': [0.05, 0.08, 0.1],
            'min_spread': [5, 7, 9],
            'volatility_spread_factor': [0.25, 0.32, 0.4],
            'inventory_skew_factor': [0.005, 0.01, 0.015],
            'reversion_threshold': [1.5, 2.0, 2.5]
        }
        
        # Define parameter grids for KELP mean reversion strategy
        kelp_grid = {
            'window_size': [30, 40, 50],
            'entry_threshold': [0.8, 1.0, 1.2],
            'exit_threshold': [0.5, 0.7, 0.9],
            'position_limit': [30, 40, 50],
            'order_size': [20, 25, 30],
            'ema_alpha': [0.03, 0.05, 0.07],
            'min_spread': [1, 2, 3],
            'volatility_spread_factor': [0.8, 1.0, 1.2],
            'inventory_skew_factor': [0.01, 0.015, 0.02],
            'min_volatility_qty_factor': [0.9, 1.0, 1.1],
            'max_volatility_for_qty_reduction': [3.0, 4.0, 5.0],
            'imbalance_depth': [3, 5, 7],
            'imbalance_fv_adjustment_factor': [0.2, 0.36, 0.5]
        }
        
        # Define parameter grids for SQUID_INK order book imbalance strategy
        squid_ink_grid = {
            'imbalance_threshold': [0.15, 0.2, 0.25],
            'take_profit': [2.0, 3.0, 4.0],
            'stop_loss': [1.5, 2.0, 2.5],
            'position_limit': [30, 40, 50],
            'order_size': [15, 20, 25],
            'ema_alpha': [0.08, 0.1, 0.12],
            'trend_strength_threshold': [0.4, 0.6, 0.8],
            'min_spread': [2, 3, 4],
            'volatility_spread_factor': [0.6, 0.8, 1.0],
            'inventory_skew_factor': [0.015, 0.02, 0.025],
            'imbalance_depth': [2, 3, 4],
            'reversal_threshold': [1.0, 1.5, 2.0]
        }
        
        # Optimize shared parameters first (optional - this is a simplified approach)
        print("\nOptimizing shared parameters (this could take a while)...")
        # For a real optimization, you'd want to test shared parameters across all products
        # Here we're just using placeholder values for demonstration
        shared_params = {
            'take_profit_threshold': 0.4,
            'max_history_length': 90
        }
        optimization_results['shared'] = shared_params
        
        # Optimize RAINFOREST_RESIN with mean reversion
        print("\nOptimizing RAINFOREST_RESIN with mean reversion strategy...")
        rainforest_result = self.grid_search('RAINFOREST_RESIN', 'mean_reversion', rainforest_resin_grid)
        optimization_results['RAINFOREST_RESIN'] = rainforest_result
        
        # Optimize KELP with mean reversion
        print("\nOptimizing KELP with mean reversion strategy...")
        kelp_result = self.grid_search('KELP', 'mean_reversion', kelp_grid)
        optimization_results['KELP'] = kelp_result
        
        # Optimize SQUID_INK with order book imbalance
        print("\nOptimizing SQUID_INK with order book imbalance strategy...")
        squid_result = self.grid_search('SQUID_INK', 'order_book_imbalance', squid_ink_grid)
        optimization_results['SQUID_INK'] = squid_result
        
        # Generate summary report
        self._generate_optimization_summary(optimization_results)
        
        return optimization_results
    
    def _generate_optimization_summary(self, optimization_results):
        """
        Generate a summary report of optimization results
        """
        # Create output directory if it doesn't exist
        os.makedirs("optimization_results", exist_ok=True)
        
        # Create summary DataFrame
        summary_data = []
        
        # Create a dictionary to store optimized parameters for the trading algorithm
        trading_params = {}
        
        for product, result in optimization_results.items():
            best_params = result['best_params']
            best_pnl = result['best_pnl']
            strategy = result['strategy']
            
            # Add to summary data
            summary_data.append({
                'Product': product,
                'Strategy': strategy,
                'Best PnL': best_pnl,
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
        summary_df.to_csv("optimization_results/optimization_summary.csv", index=False)
        
        # Generate summary text file
        with open("optimization_results/optimization_summary.txt", "w") as f:
            f.write("OPTIMIZATION SUMMARY\n")
            f.write("===================\n\n")
            
            for product, result in optimization_results.items():
                best_params = result['best_params']
                best_pnl = result['best_pnl']
                strategy = result['strategy']
                
                f.write(f"{product} - {strategy.upper()}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Best PnL: {best_pnl:.2f}\n")
                f.write("Parameters:\n")
                
                for param, value in best_params.items():
                    f.write(f"  {param}: {value}\n")
                
                f.write("\n")
        
        # Save optimization results as JSON for the trading algorithm to use
        with open("optimization_results/optimization_summary.json", "w") as f:
            json.dump(trading_params, f, indent=2)
            
        print("\nOptimization summary saved to optimization_results/optimization_summary.csv, .txt, and .json")

def main():
    # Create the optimizer
    optimizer = StrategyOptimizer(data_folder="./data")
    
    # Load data
    print("Loading data...")
    optimizer.load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    optimizer.preprocess_data()
    
    # Run optimization for all products
    print("Running optimization...")
    optimization_results = optimizer.optimize_all_products()
    
    print("\nOptimization complete!")
    print("All results and visualizations saved to the optimization_results directory")

if __name__ == "__main__":
    main()