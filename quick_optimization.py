#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import itertools
import os
from dataclasses import dataclass
import time

class QuickOptimizer:
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
        Load and preprocess market data efficiently
        """
        print("Loading data...")
        price_files = []
        trade_files = []
        
        # Specific file patterns based on day indices
        day_indices = [-2, -1, 0]
        
        # Search for data files in different locations
        for i in day_indices:
            # Locations to check in priority order
            locations = [
                self.data_folder,
                os.path.join(self.data_folder, "round-1-island-data-bottle"),
                "round-1-island-data-bottle",
                ""
            ]
            
            price_found = False
            trade_found = False
            
            for loc in locations:
                # Check price files
                price_path = os.path.join(loc, f"prices_round_1_day_{i}.csv")
                if os.path.exists(price_path) and not price_found:
                    price_files.append((i, price_path))
                    price_found = True
                    print(f"Found price data for day {i}: {price_path}")
                
                # Check trade files
                trade_path = os.path.join(loc, f"trades_round_1_day_{i}.csv")
                if os.path.exists(trade_path) and not trade_found:
                    trade_files.append((i, trade_path))
                    trade_found = True
                    print(f"Found trade data for day {i}: {trade_path}")
                
                # If both found, move to next day
                if price_found and trade_found:
                    break
        
        if not price_files:
            raise Exception("No price data files found. Please check data folder.")
        
        # Load price data
        all_price_data = []
        for day_idx, file in price_files:
            try:
                df = pd.read_csv(file, delimiter=';')
                # Ensure day column exists
                if 'day' not in df.columns:
                    df['day'] = day_idx
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
                # Add day column if not present
                if 'day' not in df.columns:
                    df['day'] = day_idx
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
        Enhanced data preprocessing with advanced features
        """
        print("Preprocessing data...")
        if self.price_data is None:
            raise Exception("No data loaded. Call load_data() first.")
        
        # Calculate mid price if not present
        if 'mid_price' not in self.price_data.columns:
            self.price_data['mid_price'] = (self.price_data['bid_price_1'] + self.price_data['ask_price_1']) / 2
        
        # Process each product
        for product in self.products:
            # Filter data for this product
            product_df = self.price_data[self.price_data['product'] == product].copy()
            product_df = product_df.sort_values(['day', 'timestamp']).reset_index(drop=True)
            
            # Calculate order book features
            product_df['total_bid_volume'] = product_df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1, skipna=True)
            product_df['total_ask_volume'] = product_df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1, skipna=True)
            
            # Order book imbalance
            product_df['imbalance_ratio'] = (
                (product_df['total_bid_volume'] - product_df['total_ask_volume']) / 
                (product_df['total_bid_volume'] + product_df['total_ask_volume'])
            ).fillna(0)
            
            # Store processed data
            self.product_data[product] = product_df
            print(f"Preprocessed {product} data: {len(product_df)} rows")
    
    @dataclass
    class TradingState:
        """Data class to track state during simulation"""
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
        base_order_qty = params.get('base_order_qty', params.get('order_size', 10))  # Support both param names
        
        # Get product data
        product_df = self.product_data[product].copy()
        day_df = product_df[product_df['day'] == day_index].copy()
        if day_df.empty:
            return {'total_pnl': 0, 'total_trades': 0}
        
        day_df = day_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate moving average
        day_df['sma'] = day_df['mid_price'].rolling(window=window_size).mean().bfill()
        day_df['deviation'] = (day_df['mid_price'] - day_df['sma']) / day_df['sma']
        
        # Initialize trading state
        state = self.TradingState()
        positions = []
        pnl = []
        
        # Start trading from when we have enough data for MA
        for i in range(window_size, len(day_df)):
            row = day_df.iloc[i]
            
            # Current price and deviation
            mid_price = row['mid_price']
            deviation = row['deviation']
            
            # Determine buy/sell actions
            action = 0  # 0: no action, 1: buy, -1: sell
            
            # If no position, check if we should enter
            if state.position == 0:
                if deviation < -entry_threshold:
                    action = 1  # buy
                elif deviation > entry_threshold:
                    action = -1  # sell
            # If we have a position, check if we should exit
            else:
                if abs(deviation) < exit_threshold:
                    action = -state.position  # exit position
            
            # Execute trade if there's an action
            quantity = 0
            if action > 0:
                # Buy, limited by position limit
                quantity = min(base_order_qty, position_limit - state.position)
                if quantity > 0:
                    state.position += quantity
                    state.cash -= quantity * mid_price
                    state.entry_price = mid_price
                    state.trades.append({'action': 'buy', 'price': mid_price, 'quantity': quantity})
            elif action < 0:
                # Sell, limited by position limit
                quantity = min(base_order_qty, position_limit + state.position)
                if quantity > 0:
                    state.position -= quantity
                    state.cash += quantity * mid_price
                    state.entry_price = mid_price if state.position != 0 else None
                    state.trades.append({'action': 'sell', 'price': mid_price, 'quantity': quantity})
            
            # Record position and PnL after each step
            positions.append(state.position)
            current_portfolio_value = state.cash + state.position * mid_price
            pnl.append(current_portfolio_value)
        
        # Calculate final portfolio value
        final_price = day_df['mid_price'].iloc[-1] if not day_df.empty else 0
        portfolio_value = state.cash + state.position * final_price
        
        # Calculate performance metrics
        total_trades = len(state.trades)
        
        # Force liquidation at the end to properly calculate PnL
        # This makes sure we account for unrealized profits/losses
        if state.position != 0:
            liquidation_value = state.position * final_price
            # Adjust PnL for the forced liquidation (this is the realistic PnL)
            portfolio_value = state.cash + liquidation_value
        
        return {
            'total_pnl': portfolio_value,
            'total_trades': total_trades,
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
        base_order_qty = params.get('base_order_qty', params.get('order_size', 10))
        
        # Get product data
        product_df = self.product_data[product].copy()
        day_df = product_df[product_df['day'] == day_index].copy()
        if day_df.empty:
            return {'total_pnl': 0, 'total_trades': 0}
        
        day_df = day_df.sort_values('timestamp').reset_index(drop=True)
        
        # Initialize trading state
        state = self.TradingState()
        positions = []
        pnl = []
        
        # Start trading
        for i in range(1, len(day_df)):
            row = day_df.iloc[i]
            
            # Current price and imbalance
            mid_price = row['mid_price']
            imbalance = row['imbalance_ratio']
            
            # Determine buy/sell actions
            action = 0  # 0: no action, 1: buy, -1: sell
            
            # Check for take profit or stop loss if in position
            if state.position != 0 and state.entry_price is not None:
                price_change_pct = (mid_price - state.entry_price) / state.entry_price * 100
                if state.position > 0:  # Long position
                    if price_change_pct >= take_profit or price_change_pct <= -stop_loss:
                        action = -1  # Close position
                else:  # Short position
                    if price_change_pct <= -take_profit or price_change_pct >= stop_loss:
                        action = 1  # Close position
            
            # If no take profit/stop loss, check if we should enter based on imbalance
            if action == 0 and state.position == 0 and abs(imbalance) > imbalance_threshold:
                if imbalance > 0:
                    action = 1  # buy
                else:
                    action = -1  # sell
            
            # Execute trade if there's an action
            quantity = 0
            if action > 0:
                # Buy, limited by position limit
                quantity = min(base_order_qty, position_limit - state.position)
                if quantity > 0:
                    state.position += quantity
                    state.cash -= quantity * mid_price
                    state.entry_price = mid_price
                    state.trades.append({'action': 'buy', 'price': mid_price, 'quantity': quantity})
            elif action < 0:
                # Sell, limited by position limit
                quantity = min(base_order_qty, position_limit + state.position)
                if quantity > 0:
                    state.position -= quantity
                    state.cash += quantity * mid_price
                    state.entry_price = mid_price if state.position != 0 else None
                    state.trades.append({'action': 'sell', 'price': mid_price, 'quantity': quantity})
            
            # Record position and PnL
            positions.append(state.position)
            current_portfolio_value = state.cash + state.position * mid_price
            pnl.append(current_portfolio_value)
        
        # Calculate final portfolio value
        final_price = day_df['mid_price'].iloc[-1] if not day_df.empty else 0
        portfolio_value = state.cash + state.position * final_price
        
        # Calculate performance metrics
        total_trades = len(state.trades)
        
        # Force liquidation at the end to properly calculate PnL
        if state.position != 0:
            liquidation_value = state.position * final_price
            # Adjust PnL for the forced liquidation
            portfolio_value = state.cash + liquidation_value
        
        return {
            'total_pnl': portfolio_value,
            'total_trades': total_trades,
            'positions': positions,
            'pnl_history': pnl
        }
    
    def quick_grid_search(self, product, strategy, param_grid, day_index=0):
        """
        Perform a quick grid search on the most recent day
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
        
        # Progress tracking
        print(f"Running grid search for {product} with {strategy} ({len(param_combinations)} combinations)...")
        start_time = time.time()
        
        # Test each parameter combination
        for i, combo in enumerate(param_combinations):
            params = dict(zip(param_keys, combo))
            
            # Run backtest with the current parameters
            if strategy == 'mean_reversion':
                result = self.backtest_mean_reversion(product, params, day_index)
            elif strategy == 'order_book_imbalance':
                result = self.backtest_order_book_imbalance(product, params, day_index)
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
                best_result = result.copy()
                best_result['params'] = params
            
            # Print progress
            if (i+1) % 100 == 0 or i+1 == len(param_combinations):
                elapsed = time.time() - start_time
                print(f"Processed {i+1}/{len(param_combinations)} combinations ({elapsed:.2f}s)")
        
        # Sort results by PnL
        sorted_results = sorted(all_results, key=lambda x: x['total_pnl'], reverse=True)
        
        # Print top results
        print("\nTop 5 parameter combinations:")
        for i, result in enumerate(sorted_results[:5]):
            params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
            print(f"{i+1}. PnL: {result['total_pnl']:.2f}, Trades: {result['total_trades']}, Params: {params_str}")
        
        return {
            'product': product,
            'strategy': strategy,
            'best_params': best_result['params'] if best_result else None,
            'best_pnl': best_pnl,
            'top_results': sorted_results[:10]
        }
    
    def optimize_all_products(self, day_index=0):
        """
        Run a quick optimization for all products
        """
        print("\nStarting optimization for all products...")
        optimization_results = {}
        
        # Shared parameters
        shared_params = {
            'take_profit_threshold': 0.4,
            'max_history_length': 90
        }
        optimization_results['shared'] = shared_params
        
        # Enhanced parameter grids for RAINFOREST_RESIN
        rainforest_resin_grid = {
            'strategy': ['mean_reversion'],
            'window_size': [20, 30, 40],
            'entry_threshold': [0.8, 1.0, 1.5],
            'exit_threshold': [0.5, 0.7, 0.9],
            'position_limit': [40, 50],
            'base_order_qty': [20, 25],
            'fair_value_anchor': [10000.0],
            'anchor_blend_alpha': [0.05, 0.08],
            'min_spread': [5, 7],
            'volatility_spread_factor': [0.28, 0.32, 0.36],
            'inventory_skew_factor': [0.008, 0.01, 0.012],
            'reversion_threshold': [1.8, 2.0, 2.2]
        }
        
        # Enhanced parameter grids for KELP
        kelp_grid = {
            'strategy': ['mean_reversion'],
            'window_size': [40, 50],
            'entry_threshold': [0.8, 1.0],
            'exit_threshold': [0.6, 0.7],
            'position_limit': [40, 50],
            'base_order_qty': [25, 28],
            'ema_alpha': [0.04, 0.05, 0.06],
            'min_spread': [2],
            'volatility_spread_factor': [1.0, 1.2],
            'inventory_skew_factor': [0.012, 0.015, 0.018],
            'min_volatility_qty_factor': [1.0, 1.1],
            'max_volatility_for_qty_reduction': [4.0],
            'imbalance_depth': [5],
            'imbalance_fv_adjustment_factor': [0.32, 0.36, 0.40]
        }
        
        # Enhanced parameter grids for SQUID_INK
        squid_ink_grid = {
            'strategy': ['order_book_imbalance'],
            'imbalance_threshold': [0.18, 0.2, 0.22],
            'take_profit': [2.5, 3.0, 3.5],
            'stop_loss': [1.8, 2.0, 2.2],
            'position_limit': [40, 50],
            'base_order_qty': [18, 20, 22],
            'ema_alpha': [0.09, 0.1, 0.11],
            'trend_strength_threshold': [0.5, 0.6, 0.7],
            'min_spread': [3],
            'volatility_spread_factor': [0.7, 0.8, 0.9],
            'inventory_skew_factor': [0.018, 0.02, 0.022],
            'imbalance_depth': [3],
            'reversal_threshold': [1.3, 1.5, 1.7]
        }
        
        # Optimize RAINFOREST_RESIN with mean reversion
        print("\nOptimizing RAINFOREST_RESIN with mean reversion strategy...")
        rainforest_result = self.quick_grid_search('RAINFOREST_RESIN', 'mean_reversion', rainforest_resin_grid, day_index)
        optimization_results['RAINFOREST_RESIN'] = rainforest_result
        
        # Optimize KELP with mean reversion
        print("\nOptimizing KELP with mean reversion strategy...")
        kelp_result = self.quick_grid_search('KELP', 'mean_reversion', kelp_grid, day_index)
        optimization_results['KELP'] = kelp_result
        
        # Optimize SQUID_INK with order book imbalance
        print("\nOptimizing SQUID_INK with order book imbalance strategy...")
        squid_result = self.quick_grid_search('SQUID_INK', 'order_book_imbalance', squid_ink_grid, day_index)
        optimization_results['SQUID_INK'] = squid_result
        
        # Create output directory if it doesn't exist
        os.makedirs("optimization_results", exist_ok=True)
        
        # Create a dictionary to store optimized parameters for the trading algorithm
        trading_params = {}
        
        # Print overall summary
        print("\nOptimization Summary:")
        for product, result in optimization_results.items():
            best_params = result['best_params']
            best_pnl = result['best_pnl']
            strategy = result['strategy']
            
            # Add to trading parameters
            trading_params[product] = {
                'strategy': strategy,
                **best_params
            }
            
            print(f"\n{product} - {strategy.upper()}")
            print("-" * 50)
            print(f"Best PnL: {best_pnl:.2f}")
            print("Parameters:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
        
        # Save optimization results as JSON for the trading algorithm to use
        with open("optimization_results/optimization_summary.json", "w") as f:
            json.dump(trading_params, f, indent=2)
            
        print("\nOptimization summary saved to optimization_results/optimization_summary.json")
        
        return optimization_results

def main():
    # Create the optimizer
    optimizer = QuickOptimizer(data_folder="./data")
    
    # Load data
    optimizer.load_data()
    
    # Run optimization for all products on the most recent day (day 0)
    optimization_results = optimizer.optimize_all_products(day_index=0)
    
    print("\nOptimization complete!")

if __name__ == "__main__":
    main()