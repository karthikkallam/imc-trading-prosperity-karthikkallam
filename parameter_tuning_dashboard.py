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
from optimization_framework import StrategyOptimizer

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
        
        # Parameter configuration based on strategy
        st.sidebar.subheader("Strategy Parameters")
        
        params = {}
        
        if strategy_internal == "mean_reversion":
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                params['window_size'] = st.number_input(
                    "Moving Average Window Size", 
                    min_value=5, 
                    max_value=100, 
                    value=30,
                    step=5
                )
                
                params['entry_threshold'] = st.number_input(
                    "Entry Threshold", 
                    min_value=0.5, 
                    max_value=5.0, 
                    value=2.0,
                    step=0.5
                )
                
                params['position_limit'] = st.number_input(
                    "Position Limit", 
                    min_value=5, 
                    max_value=50, 
                    value=20,
                    step=5
                )
            
            with col2:
                params['exit_threshold'] = st.number_input(
                    "Exit Threshold", 
                    min_value=0.1, 
                    max_value=3.0, 
                    value=0.5,
                    step=0.1
                )
                
                params['order_size'] = st.number_input(
                    "Order Size", 
                    min_value=1, 
                    max_value=20, 
                    value=5,
                    step=1
                )
        
        elif strategy_internal == "order_book_imbalance":
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                params['imbalance_threshold'] = st.number_input(
                    "Imbalance Threshold", 
                    min_value=0.1, 
                    max_value=0.9, 
                    value=0.5,
                    step=0.1
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
                    value=20,
                    step=5
                )
            
            with col2:
                params['stop_loss'] = st.number_input(
                    "Stop Loss", 
                    min_value=0.5, 
                    max_value=5.0, 
                    value=1.5,
                    step=0.5
                )
                
                params['order_size'] = st.number_input(
                    "Order Size", 
                    min_value=1, 
                    max_value=20, 
                    value=5,
                    step=1
                )
        
        # Run backtest button
        if st.sidebar.button("Run Backtest"):
            with st.spinner(f"Running {strategy} strategy backtest for {product}..."):
                try:
                    start_time = time.time()
                    
                    if strategy_internal == "mean_reversion":
                        result = optimizer.backtest_mean_reversion(product, params, visualize=True)
                    elif strategy_internal == "order_book_imbalance":
                        result = optimizer.backtest_order_book_imbalance(product, params, visualize=True)
                    
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
                    # Define parameter grid based on strategy
                    if strategy_internal == "mean_reversion":
                        param_grid = {
                            'window_size': [params['window_size'] - 10, params['window_size'], params['window_size'] + 10],
                            'entry_threshold': [params['entry_threshold'] - 0.5, params['entry_threshold'], params['entry_threshold'] + 0.5],
                            'exit_threshold': [params['exit_threshold'] - 0.1, params['exit_threshold'], params['exit_threshold'] + 0.1],
                            'position_limit': [params['position_limit']],
                            'order_size': [params['order_size'] - 1, params['order_size'], params['order_size'] + 1]
                        }
                    elif strategy_internal == "order_book_imbalance":
                        param_grid = {
                            'imbalance_threshold': [params['imbalance_threshold'] - 0.1, params['imbalance_threshold'], params['imbalance_threshold'] + 0.1],
                            'take_profit': [params['take_profit'] - 0.5, params['take_profit'], params['take_profit'] + 0.5],
                            'stop_loss': [params['stop_loss'] - 0.5, params['stop_loss'], params['stop_loss'] + 0.5],
                            'position_limit': [params['position_limit']],
                            'order_size': [params['order_size'] - 1, params['order_size'], params['order_size'] + 1]
                        }
                    
                    # Clean up negative values
                    for key, values in param_grid.items():
                        param_grid[key] = [max(0.1, v) for v in values]
                    
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