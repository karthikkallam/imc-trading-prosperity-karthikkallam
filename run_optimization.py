#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import time
import subprocess
import sys

def ensure_dependencies():
    """Ensure all required dependencies are installed"""
    try:
        import scipy
        import sklearn
    except ImportError:
        print("Installing required dependencies for Bayesian Optimization...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "scikit-learn"])
        print("Dependencies installed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Run trading strategy optimization')
    parser.add_argument('--quick', action='store_true', help='Use quick optimization (faster but less thorough)')
    parser.add_argument('--bayesian', action='store_true', help='Use Bayesian optimization (most effective)')
    parser.add_argument('--dashboard', action='store_true', help='Launch the parameter tuning dashboard')
    parser.add_argument('--data-folder', type=str, default='./data', help='Path to data folder')
    parser.add_argument('--iterations', type=int, default=None, help='Number of optimization iterations')
    parser.add_argument('--product', type=str, default=None, help='Optimize only a specific product')
    args = parser.parse_args()
    
    # Create optimization_results directory if it doesn't exist
    os.makedirs("optimization_results", exist_ok=True)
    
    if args.dashboard:
        print("Launching parameter tuning dashboard...")
        os.system("python parameter_tuning_dashboard.py")
        return
    
    start_time = time.time()
    
    if args.bayesian:
        print("Running Bayesian optimization (most effective but requires additional dependencies)...")
        ensure_dependencies()  # Ensure required packages are installed
        
        # Import here to avoid importing unnecessary modules when not needed
        from bayesian_optimization import BayesianOptimizer
        
        optimizer = BayesianOptimizer(data_folder=args.data_folder)
        optimizer.load_data()
        
        # Determine iterations
        iterations = args.iterations or 30  # Default to 30 for Bayesian optimization
        
        if args.product:
            if args.product == 'RAINFOREST_RESIN' or args.product == 'KELP':
                optimizer.optimize_mean_reversion(args.product, day_index=0, n_iterations=iterations)
            elif args.product == 'SQUID_INK':
                optimizer.optimize_order_book_imbalance(args.product, day_index=0, n_iterations=iterations)
            else:
                print(f"Unknown product: {args.product}")
        else:
            optimizer.optimize_all_products(day_index=0, n_iterations=iterations)
    
    elif args.quick:
        print("Running quick optimization...")
        # Import here to avoid importing unnecessary modules when not needed
        from quick_optimization import QuickOptimizer
        
        optimizer = QuickOptimizer(data_folder=args.data_folder)
        optimizer.load_data()
        
        if args.product:
            # Quick grid search for a specific product
            if args.product == 'RAINFOREST_RESIN' or args.product == 'KELP':
                # Define minimal parameter grid for mean reversion
                param_grid = {
                    'window_size': [20, 30, 40],
                    'entry_threshold': [0.8, 1.0, 1.2],
                    'exit_threshold': [0.5, 0.7],
                    'position_limit': [50],
                    'base_order_qty': [20, 25]
                }
                optimizer.quick_grid_search(args.product, 'mean_reversion', param_grid)
            elif args.product == 'SQUID_INK':
                # Define minimal parameter grid for order book imbalance
                param_grid = {
                    'imbalance_threshold': [0.15, 0.2, 0.25],
                    'take_profit': [2.0, 3.0, 4.0],
                    'stop_loss': [1.5, 2.0, 2.5],
                    'position_limit': [50],
                    'base_order_qty': [15, 20, 25]
                }
                optimizer.quick_grid_search(args.product, 'order_book_imbalance', param_grid)
            else:
                print(f"Unknown product: {args.product}")
        else:
            optimizer.optimize_all_products(day_index=0)
    else:
        print("Running full grid optimization (this may take some time)...")
        # Import here to avoid importing unnecessary modules when not needed
        from optimization_framework import StrategyOptimizer
        
        optimizer = StrategyOptimizer(data_folder=args.data_folder)
        optimizer.load_data()
        optimizer.preprocess_data()
        
        if args.product:
            # Create parameter grid for the specific product
            if args.product == 'RAINFOREST_RESIN' or args.product == 'KELP':
                strategy = 'mean_reversion'
                param_grid = {
                    'window_size': [20, 30, 40],
                    'entry_threshold': [0.8, 1.0, 1.2],
                    'exit_threshold': [0.5, 0.7, 0.9],
                    'position_limit': [40, 50],
                    'order_size': [20, 25]
                }
                optimizer.grid_search(args.product, strategy, param_grid, True)
            elif args.product == 'SQUID_INK':
                strategy = 'order_book_imbalance'
                param_grid = {
                    'imbalance_threshold': [0.15, 0.2, 0.25],
                    'take_profit': [2.0, 3.0, 4.0],
                    'stop_loss': [1.5, 2.0, 2.5],
                    'position_limit': [40, 50],
                    'order_size': [15, 20, 25]
                }
                optimizer.grid_search(args.product, strategy, param_grid, True)
            else:
                print(f"Unknown product: {args.product}")
        else:
            optimizer.optimize_all_products()
    
    elapsed_time = time.time() - start_time
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
    print("\nYou can now use the optimized parameters in the trading algorithm:")
    print("1. The parameters have been saved to optimization_results/optimization_summary.json")
    print("2. The trading_algorithm.py file will automatically load these parameters")
    print("3. For interactive tuning, run: python run_optimization.py --dashboard")

if __name__ == "__main__":
    main()