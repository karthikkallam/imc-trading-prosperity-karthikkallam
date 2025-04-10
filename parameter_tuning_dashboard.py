#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, List, Tuple
import time

# Import the strategy optimizer
# You can use either the full optimizer or the quick optimizer
from optimization_framework import StrategyOptimizer
# from quick_optimization import QuickOptimizer as StrategyOptimizer  # Uncomment to use quick optimizer

# Set page config
st.set_page_config(
    page_title="IMC Prosperity Trading Strategy Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions to create the dashboard
def create_dashboard():
    """
    Create the Streamlit dashboard for parameter tuning
    """
    st.title("IMC Prosperity Trading Strategy Optimizer")
    
    st.sidebar.header("Settings")
    
    # Data loading section
    st.sidebar.subheader("Data")
    
    # Automatically detect possible data folders
    possible_folders = ["./data", "./round-1-island-data-bottle", "./"]
    valid_folders = [folder for folder in possible_folders if os.path.exists(folder)]
    
    data_folder = st.sidebar.selectbox(
        "Data Folder Path",
        options=valid_folders,
        index=0 if valid_folders else None
    )
    
    # Initialize optimizer
    optimizer = None
    
    # Load data button
    if st.sidebar.button("Load Data"):
        with st.spinner("Loading and preprocessing data..."):
            try:
                optimizer = StrategyOptimizer(data_folder=data_folder)
                optimizer.load_data()
                optimizer.preprocess_data()
                st.session_state['optimizer'] = optimizer
                st.sidebar.success("Data loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading data: {str(e)}")
                st.error(f"Detailed error: {str(e)}")
    
    # Check if optimizer exists in session state
    if 'optimizer' in st.session_state:
        optimizer = st.session_state['optimizer']
        
        # Strategy selection
        st.sidebar.subheader("Strategy Configuration")
        product = st.sidebar.selectbox(
            "Select Product", 
            ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]
        )
        
        strategy_map = {
            "RAINFOREST_RESIN": ["Mean Reversion", "Order Book Imbalance"],
            "KELP": ["Mean Reversion", "Order Book Imbalance"],
            "SQUID_INK": ["Order Book Imbalance", "Mean Reversion"]
        }
        
        strategy = st.sidebar.selectbox(
            "Select Strategy",
            strategy_map[product]
        )
        
        # Convert strategy display name to internal name
        strategy_internal = strategy.lower().replace(" ", "_")
        
        # Shared parameters across all strategies
        st.sidebar.subheader("Shared Parameters")
        shared_params = {}
        
        # Create expander for shared parameters
        with st.sidebar.expander("Shared Parameters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                shared_params['take_profit_threshold'] = st.number_input(
                    "Take Profit Threshold", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.4,
                    step=0.1
                )
            
            with col2:
                shared_params['max_history_length'] = st.number_input(
                    "Max History Length", 
                    min_value=30, 
                    max_value=200, 
                    value=90,
                    step=10
                )
        
        # Parameter configuration based on strategy
        st.sidebar.subheader("Strategy Parameters")
        
        params = {}
        
        if strategy_internal == "mean_reversion":
            # Create tabs for different parameter categories
            basic_tab, advanced_tab = st.sidebar.tabs(["Basic", "Advanced"])
            
            with basic_tab:
                col1, col2 = st.columns(2)
                
                with col1:
                    params['window_size'] = st.number_input(
                        "MA Window Size", 
                        min_value=5, 
                        max_value=100, 
                        value=30,
                        step=5
                    )
                    
                    params['entry_threshold'] = st.number_input(
                        "Entry Threshold", 
                        min_value=0.5, 
                        max_value=5.0, 
                        value=1.0,
                        step=0.1
                    )
                    
                    params['position_limit'] = st.number_input(
                        "Position Limit", 
                        min_value=5, 
                        max_value=50, 
                        value=40,
                        step=5
                    )
                
                with col2:
                    params['exit_threshold'] = st.number_input(
                        "Exit Threshold", 
                        min_value=0.1, 
                        max_value=3.0, 
                        value=0.7,
                        step=0.1
                    )
                    
                    params['base_order_qty'] = st.number_input(
                        "Base Order Qty", 
                        min_value=1, 
                        max_value=50, 
                        value=25,
                        step=1
                    )
            
            with advanced_tab:
                # Different advanced parameters based on product
                if product == "RAINFOREST_RESIN":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        params['fair_value_anchor'] = st.number_input(
                            "Fair Value Anchor", 
                            min_value=9900.0, 
                            max_value=10100.0, 
                            value=10000.0,
                            step=10.0
                        )
                        
                        params['anchor_blend_alpha'] = st.number_input(
                            "Anchor Blend Alpha", 
                            min_value=0.01, 
                            max_value=0.5, 
                            value=0.08,
                            step=0.01
                        )
                        
                        params['min_spread'] = st.number_input(
                            "Min Spread", 
                            min_value=1, 
                            max_value=20, 
                            value=7,
                            step=1
                        )
                    
                    with col2:
                        params['volatility_spread_factor'] = st.number_input(
                            "Volatility Spread Factor", 
                            min_value=0.1, 
                            max_value=1.0, 
                            value=0.32,
                            step=0.02
                        )
                        
                        params['inventory_skew_factor'] = st.number_input(
                            "Inventory Skew Factor", 
                            min_value=0.001, 
                            max_value=0.05, 
                            value=0.01,
                            step=0.001
                        )
                        
                        params['reversion_threshold'] = st.number_input(
                            "Reversion Threshold", 
                            min_value=0.5, 
                            max_value=5.0, 
                            value=2.0,
                            step=0.1
                        )
                
                elif product == "KELP":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        params['ema_alpha'] = st.number_input(
                            "EMA Alpha", 
                            min_value=0.01, 
                            max_value=0.5, 
                            value=0.05,
                            step=0.01
                        )
                        
                        params['min_spread'] = st.number_input(
                            "Min Spread", 
                            min_value=1, 
                            max_value=10, 
                            value=2,
                            step=1
                        )
                        
                        params['volatility_spread_factor'] = st.number_input(
                            "Volatility Spread Factor", 
                            min_value=0.5, 
                            max_value=2.0, 
                            value=1.2,
                            step=0.1
                        )
                        
                        params['inventory_skew_factor'] = st.number_input(
                            "Inventory Skew Factor", 
                            min_value=0.005, 
                            max_value=0.05, 
                            value=0.015,
                            step=0.001
                        )
                    
                    with col2:
                        params['min_volatility_qty_factor'] = st.number_input(
                            "Min Vol Qty Factor", 
                            min_value=0.5, 
                            max_value=1.5, 
                            value=1.1,
                            step=0.1
                        )
                        
                        params['max_volatility_for_qty_reduction'] = st.number_input(
                            "Max Vol for Qty Reduction", 
                            min_value=1.0, 
                            max_value=10.0, 
                            value=4.0,
                            step=0.5
                        )
                        
                        params['imbalance_depth'] = st.number_input(
                            "Imbalance Depth", 
                            min_value=1, 
                            max_value=10, 
                            value=5,
                            step=1
                        )
                        
                        params['imbalance_fv_adjustment_factor'] = st.number_input(
                            "Imbalance FV Adj Factor", 
                            min_value=0.1, 
                            max_value=1.0, 
                            value=0.36,
                            step=0.02
                        )
        
        elif strategy_internal == "order_book_imbalance":
            # Create tabs for different parameter categories
            basic_tab, advanced_tab = st.sidebar.tabs(["Basic", "Advanced"])
            
            with basic_tab:
                col1, col2 = st.columns(2)
                
                with col1:
                    params['imbalance_threshold'] = st.number_input(
                        "Imbalance Threshold", 
                        min_value=0.1, 
                        max_value=0.9, 
                        value=0.2,
                        step=0.02
                    )
                    
                    params['take_profit'] = st.number_input(
                        "Take Profit", 
                        min_value=0.5, 
                        max_value=10.0, 
                        value=3.0,
                        step=0.5
                    )
                    
                    params['position_limit'] = st.number_input(
                        "Position Limit", 
                        min_value=5, 
                        max_value=50, 
                        value=40,
                        step=5
                    )
                
                with col2:
                    params['stop_loss'] = st.number_input(
                        "Stop Loss", 
                        min_value=0.5, 
                        max_value=5.0, 
                        value=2.0,
                        step=0.1
                    )
                    
                    params['base_order_qty'] = st.number_input(
                        "Base Order Qty", 
                        min_value=1, 
                        max_value=50, 
                        value=20,
                        step=1
                    )
            
            with advanced_tab:
                col1, col2 = st.columns(2)
                
                with col1:
                    params['ema_alpha'] = st.number_input(
                        "EMA Alpha", 
                        min_value=0.01, 
                        max_value=0.5, 
                        value=0.1,
                        step=0.01
                    )
                    
                    params['trend_strength_threshold'] = st.number_input(
                        "Trend Strength Threshold", 
                        min_value=0.2, 
                        max_value=1.0, 
                        value=0.6,
                        step=0.1
                    )
                    
                    params['min_spread'] = st.number_input(
                        "Min Spread", 
                        min_value=1, 
                        max_value=10, 
                        value=3,
                        step=1
                    )
                    
                    params['volatility_spread_factor'] = st.number_input(
                        "Volatility Spread Factor", 
                        min_value=0.1, 
                        max_value=2.0, 
                        value=0.8,
                        step=0.1
                    )
                
                with col2:
                    params['inventory_skew_factor'] = st.number_input(
                        "Inventory Skew Factor", 
                        min_value=0.005, 
                        max_value=0.05, 
                        value=0.02,
                        step=0.005
                    )
                    
                    params['imbalance_depth'] = st.number_input(
                        "Imbalance Depth", 
                        min_value=1, 
                        max_value=10, 
                        value=3,
                        step=1
                    )
                    
                    params['reversal_threshold'] = st.number_input(
                        "Reversal Threshold", 
                        min_value=0.5, 
                        max_value=3.0, 
                        value=1.5,
                        step=0.1
                    )
        
        # Run backtest button
        if st.sidebar.button("Run Backtest"):
            with st.spinner(f"Running {strategy} strategy backtest for {product}..."):
                try:
                    start_time = time.time()
                    
                    # Combine shared parameters with strategy-specific parameters
                    combined_params = {**shared_params, **params}
                    
                    if strategy_internal == "mean_reversion":
                        result = optimizer.backtest_mean_reversion(product, combined_params, visualize=True)
                    elif strategy_internal == "order_book_imbalance":
                        result = optimizer.backtest_order_book_imbalance(product, combined_params, visualize=True)
                    
                    st.session_state['backtest_result'] = result
                    st.session_state['backtest_time'] = time.time() - start_time
                    
                    st.sidebar.success(f"Backtest completed in {st.session_state['backtest_time']:.2f} seconds!")
                except Exception as e:
                    st.sidebar.error(f"Error running backtest: {str(e)}")
                    st.error(f"Detailed error: {str(e)}")
        
        # Run grid search button
        if st.sidebar.button("Run Grid Search"):
            with st.spinner(f"Running grid search for {product} with {strategy} strategy..."):
                try:
                    # Define grid search ranges based on current parameters
                    param_grid = {}
                    
                    # Add shared parameters to the grid
                    param_grid['take_profit_threshold'] = [
                        shared_params['take_profit_threshold'] - 0.1,
                        shared_params['take_profit_threshold'],
                        shared_params['take_profit_threshold'] + 0.1
                    ]
                    
                    param_grid['max_history_length'] = [shared_params['max_history_length']]  # Keep this fixed
                    
                    # Add strategy-specific parameters to the grid
                    if strategy_internal == "mean_reversion":
                        # Basic parameters
                        param_grid['window_size'] = [params['window_size'] - 5, params['window_size'], params['window_size'] + 5]
                        param_grid['entry_threshold'] = [params['entry_threshold'] - 0.2, params['entry_threshold'], params['entry_threshold'] + 0.2]
                        param_grid['exit_threshold'] = [params['exit_threshold'] - 0.1, params['exit_threshold'], params['exit_threshold'] + 0.1]
                        param_grid['position_limit'] = [params['position_limit']]
                        param_grid['base_order_qty'] = [params['base_order_qty'] - 2, params['base_order_qty'], params['base_order_qty'] + 2]
                        
                        # Advanced parameters depend on product
                        if product == "RAINFOREST_RESIN":
                            if 'fair_value_anchor' in params:
                                param_grid['fair_value_anchor'] = [params['fair_value_anchor']]  # Keep fixed
                            
                            if 'anchor_blend_alpha' in params:
                                param_grid['anchor_blend_alpha'] = [
                                    max(0.01, params['anchor_blend_alpha'] - 0.02),
                                    params['anchor_blend_alpha'],
                                    params['anchor_blend_alpha'] + 0.02
                                ]
                            
                            if 'min_spread' in params:
                                param_grid['min_spread'] = [params['min_spread']]  # Keep fixed
                            
                            if 'volatility_spread_factor' in params:
                                param_grid['volatility_spread_factor'] = [
                                    max(0.1, params['volatility_spread_factor'] - 0.05),
                                    params['volatility_spread_factor'],
                                    params['volatility_spread_factor'] + 0.05
                                ]
                            
                            if 'inventory_skew_factor' in params:
                                param_grid['inventory_skew_factor'] = [
                                    max(0.001, params['inventory_skew_factor'] - 0.002),
                                    params['inventory_skew_factor'],
                                    params['inventory_skew_factor'] + 0.002
                                ]
                            
                            if 'reversion_threshold' in params:
                                param_grid['reversion_threshold'] = [
                                    max(0.5, params['reversion_threshold'] - 0.2),
                                    params['reversion_threshold'],
                                    params['reversion_threshold'] + 0.2
                                ]
                                
                        elif product == "KELP":
                            # Add KELP-specific parameters if used
                            if 'ema_alpha' in params:
                                param_grid['ema_alpha'] = [
                                    max(0.01, params['ema_alpha'] - 0.01),
                                    params['ema_alpha'],
                                    params['ema_alpha'] + 0.01
                                ]
                            
                            if 'min_spread' in params:
                                param_grid['min_spread'] = [params['min_spread']]  # Keep fixed
                                
                            if 'volatility_spread_factor' in params:
                                param_grid['volatility_spread_factor'] = [
                                    max(0.1, params['volatility_spread_factor'] - 0.1),
                                    params['volatility_spread_factor'],
                                    params['volatility_spread_factor'] + 0.1
                                ]
                                
                            if 'inventory_skew_factor' in params:
                                param_grid['inventory_skew_factor'] = [
                                    max(0.001, params['inventory_skew_factor'] - 0.002),
                                    params['inventory_skew_factor'],
                                    params['inventory_skew_factor'] + 0.002
                                ]
                                
                            if 'min_volatility_qty_factor' in params:
                                param_grid['min_volatility_qty_factor'] = [params['min_volatility_qty_factor']]  # Keep fixed
                                
                            if 'max_volatility_for_qty_reduction' in params:
                                param_grid['max_volatility_for_qty_reduction'] = [params['max_volatility_for_qty_reduction']]  # Keep fixed
                                
                            if 'imbalance_depth' in params:
                                param_grid['imbalance_depth'] = [params['imbalance_depth']]  # Keep fixed
                                
                            if 'imbalance_fv_adjustment_factor' in params:
                                param_grid['imbalance_fv_adjustment_factor'] = [
                                    max(0.1, params['imbalance_fv_adjustment_factor'] - 0.04),
                                    params['imbalance_fv_adjustment_factor'],
                                    params['imbalance_fv_adjustment_factor'] + 0.04
                                ]
                        
                    elif strategy_internal == "order_book_imbalance":
                        # Basic parameters
                        param_grid['imbalance_threshold'] = [
                            max(0.05, params['imbalance_threshold'] - 0.02),
                            params['imbalance_threshold'],
                            params['imbalance_threshold'] + 0.02
                        ]
                        
                        param_grid['take_profit'] = [
                            max(0.5, params['take_profit'] - 0.5),
                            params['take_profit'],
                            params['take_profit'] + 0.5
                        ]
                        
                        param_grid['stop_loss'] = [
                            max(0.5, params['stop_loss'] - 0.2),
                            params['stop_loss'],
                            params['stop_loss'] + 0.2
                        ]
                        
                        param_grid['position_limit'] = [params['position_limit']]
                        param_grid['base_order_qty'] = [
                            max(1, params['base_order_qty'] - 2),
                            params['base_order_qty'],
                            params['base_order_qty'] + 2
                        ]
                        
                        # Advanced parameters
                        if 'ema_alpha' in params:
                            param_grid['ema_alpha'] = [
                                max(0.01, params['ema_alpha'] - 0.01),
                                params['ema_alpha'],
                                params['ema_alpha'] + 0.01
                            ]
                            
                        if 'trend_strength_threshold' in params:
                            param_grid['trend_strength_threshold'] = [
                                max(0.1, params['trend_strength_threshold'] - 0.1),
                                params['trend_strength_threshold'],
                                params['trend_strength_threshold'] + 0.1
                            ]
                            
                        if 'min_spread' in params:
                            param_grid['min_spread'] = [params['min_spread']]  # Keep fixed
                            
                        if 'volatility_spread_factor' in params:
                            param_grid['volatility_spread_factor'] = [
                                max(0.1, params['volatility_spread_factor'] - 0.1),
                                params['volatility_spread_factor'],
                                params['volatility_spread_factor'] + 0.1
                            ]
                            
                        if 'inventory_skew_factor' in params:
                            param_grid['inventory_skew_factor'] = [
                                max(0.005, params['inventory_skew_factor'] - 0.005),
                                params['inventory_skew_factor'],
                                params['inventory_skew_factor'] + 0.005
                            ]
                            
                        if 'imbalance_depth' in params:
                            param_grid['imbalance_depth'] = [params['imbalance_depth']]  # Keep fixed
                            
                        if 'reversal_threshold' in params:
                            param_grid['reversal_threshold'] = [
                                max(0.5, params['reversal_threshold'] - 0.2),
                                params['reversal_threshold'],
                                params['reversal_threshold'] + 0.2
                            ]
                    
                    # Make sure all parameter values are valid
                    for key, values in param_grid.items():
                        if key in ['take_profit_threshold', 'imbalance_threshold', 'entry_threshold', 'exit_threshold', 'ema_alpha']:
                            param_grid[key] = [max(0.01, v) for v in values]
                        elif key in ['take_profit', 'stop_loss', 'reversion_threshold', 'reversal_threshold']:
                            param_grid[key] = [max(0.1, v) for v in values]
                    
                    # Add strategy to grid
                    param_grid['strategy'] = [strategy_internal]
                    
                    start_time = time.time()
                    result = optimizer.grid_search(product, strategy_internal, param_grid, visualize_best=True)
                    st.session_state['grid_search_result'] = result
                    st.session_state['grid_search_time'] = time.time() - start_time
                    
                    st.sidebar.success(f"Grid search completed in {st.session_state['grid_search_time']:.2f} seconds!")
                except Exception as e:
                    st.sidebar.error(f"Error running grid search: {str(e)}")
                    st.error(f"Detailed error: {str(e)}")
        
        # Main content area
        st.header(f"{product} - {strategy} Strategy")
        
        # If we have a backtest result, display it
        if 'backtest_result' in st.session_state:
            result = st.session_state['backtest_result']
            
            # Only display if product and strategy match
            if result['strategy'] == strategy_internal and result['params'] == params:
                st.subheader("Backtest Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total PnL", f"{result['total_pnl']:.2f}")
                col2.metric("Total Trades", f"{result['total_trades']}")
                if result['total_trades'] > 0:
                    col3.metric("Avg PnL per Trade", f"{result['total_pnl'] / result['total_trades']:.2f}")
                
                # Results by day
                st.subheader("Results by Day")
                
                day_metrics = []
                for day, day_result in result['results_by_day'].items():
                    day_metrics.append({
                        "Day": day,
                        "PnL": day_result['final_pnl'],
                        "Trades": day_result['total_trades'],
                        "Max Position": day_result['max_position']
                    })
                
                day_df = pd.DataFrame(day_metrics)
                st.table(day_df)
                
                # Display PnL chart
                st.subheader("PnL Charts")
                
                for day, day_result in result['results_by_day'].items():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(day_result['timestamps'], day_result['pnl'])
                    ax.set_title(f"Day {day} - PnL over Time")
                    ax.set_xlabel("Timestamp")
                    ax.set_ylabel("PnL")
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Display position chart
                st.subheader("Position Charts")
                
                for day, day_result in result['results_by_day'].items():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(day_result['timestamps'], day_result['positions'])
                    ax.set_title(f"Day {day} - Position over Time")
                    ax.set_xlabel("Timestamp")
                    ax.set_ylabel("Position")
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.close(fig)
        
        # If we have a grid search result, display it
        if 'grid_search_result' in st.session_state:
            result = st.session_state['grid_search_result']
            
            # Only display if product and strategy match
            if result['product'] == product and result['strategy'] == strategy_internal:
                st.subheader("Grid Search Results")
                
                # Best parameters
                st.write("Best Parameters:")
                st.json(result['best_params'])
                
                st.metric("Best PnL", f"{result['best_pnl']:.2f}")
                
                # Display parameter impact charts
                st.subheader("Parameter Impact")
                
                # Check if optimization results directory exists
                if os.path.exists("optimization_results"):
                    # Look for parameter impact images
                    for param in result['best_params'].keys():
                        impact_file = f"optimization_results/{product}_{strategy_internal}_{param}_impact.png"
                        if os.path.exists(impact_file):
                            st.image(impact_file, caption=f"Impact of {param} on PnL")
                
                # Top results
                st.subheader("Top Parameter Combinations")
                
                # Convert all results to DataFrame
                all_results_df = pd.DataFrame([
                    {**r['params'], 'total_pnl': r['total_pnl'], 'total_trades': r['total_trades']}
                    for r in result['all_results']
                ])
                
                # Display top 10 results
                top_results = all_results_df.sort_values('total_pnl', ascending=False).head(10)
                st.table(top_results)
    
    else:
        st.info("Please load data first using the button in the sidebar.")
        
        # Show example results
        st.header("Example Results")
        
        # Create sample visualization for demonstration
        try:
            # Create a simple sample chart to display
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y)
            ax.set_title("Sample Visualization (Will be replaced with actual data)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
        except:
            # Fallback to text if matplotlib fails
            st.write("Visualization will appear here after loading data")
        
        st.markdown("""
        This dashboard allows you to:
        
        1. **Load and preprocess** historical market data
        2. **Configure and backtest** different trading strategies
        3. **Optimize parameters** using grid search
        4. **Visualize results** with interactive charts
        
        To get started, click on "Load Data" in the sidebar.
        """)
if __name__ == "__main__":
    create_dashboard()