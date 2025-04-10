#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from quick_optimization import QuickOptimizer
import os
import json

def main():
    # Create the optimizer
    optimizer = QuickOptimizer(data_folder="./data")
    
    # Load data
    optimizer.load_data()
    
    # Define a very small parameter grid for testing
    tiny_resin_grid = {
        'window_size': [30],
        'entry_threshold': [1.0],
        'exit_threshold': [0.7],
        'position_limit': [50],
        'base_order_qty': [20]
    }
    
    tiny_kelp_grid = {
        'window_size': [40],
        'entry_threshold': [0.8],
        'exit_threshold': [0.6],
        'position_limit': [50],
        'base_order_qty': [25]
    }
    
    tiny_squid_grid = {
        'imbalance_threshold': [0.2],
        'take_profit': [3.0],
        'stop_loss': [2.0],
        'position_limit': [50],
        'base_order_qty': [20]
    }
    
    # Run test optimizations on all products
    print("\nTesting optimization with small parameter grids...")
    results = {}
    
    # Run for RAINFOREST_RESIN
    print("\nTesting RAINFOREST_RESIN with mean reversion strategy...")
    resin_result = optimizer.quick_grid_search('RAINFOREST_RESIN', 'mean_reversion', tiny_resin_grid, day_index=0)
    results['RAINFOREST_RESIN'] = {
        'strategy': 'mean_reversion',
        **resin_result['best_params']
    }
    
    # Run for KELP
    print("\nTesting KELP with mean reversion strategy...")
    kelp_result = optimizer.quick_grid_search('KELP', 'mean_reversion', tiny_kelp_grid, day_index=0)
    results['KELP'] = {
        'strategy': 'mean_reversion',
        **kelp_result['best_params']
    }
    
    # Run for SQUID_INK
    print("\nTesting SQUID_INK with order book imbalance strategy...")
    squid_result = optimizer.quick_grid_search('SQUID_INK', 'order_book_imbalance', tiny_squid_grid, day_index=0)
    results['SQUID_INK'] = {
        'strategy': 'order_book_imbalance',
        **squid_result['best_params']
    }
    
    # Create output directory if it doesn't exist
    os.makedirs("optimization_results", exist_ok=True)
    
    # Add shared parameters
    results['shared'] = {
        'take_profit_threshold': 0.4,
        'max_history_length': 90
    }
    
    # Save optimization results as JSON for the trading algorithm to use
    with open("optimization_results/optimization_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nTest optimization complete!")
    print(f"Optimization results saved to optimization_results/optimization_summary.json")
    print("\nIf no errors occurred, the optimization framework is working correctly!")

if __name__ == "__main__":
    main()