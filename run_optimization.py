#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Run trading strategy optimization')
    parser.add_argument('--quick', action='store_true', help='Use quick optimization (faster but less thorough)')
    parser.add_argument('--dashboard', action='store_true', help='Launch the parameter tuning dashboard')
    parser.add_argument('--data-folder', type=str, default='./data', help='Path to data folder')
    args = parser.parse_args()
    
    # Create optimization_results directory if it doesn't exist
    os.makedirs("optimization_results", exist_ok=True)
    
    if args.dashboard:
        print("Launching parameter tuning dashboard...")
        os.system("python parameter_tuning_dashboard.py")
        return
    
    start_time = time.time()
    
    if args.quick:
        print("Running quick optimization...")
        # Import here to avoid importing unnecessary modules when not needed
        from quick_optimization import QuickOptimizer
        
        optimizer = QuickOptimizer(data_folder=args.data_folder)
        optimizer.load_data()
        optimizer.optimize_all_products(day_index=0)
    else:
        print("Running full optimization (this may take some time)...")
        # Import here to avoid importing unnecessary modules when not needed
        from optimization_framework import StrategyOptimizer
        
        optimizer = StrategyOptimizer(data_folder=args.data_folder)
        optimizer.load_data()
        optimizer.preprocess_data()
        optimizer.optimize_all_products()
    
    elapsed_time = time.time() - start_time
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
    print("\nYou can now use the optimized parameters in the trading algorithm:")
    print("1. The parameters have been saved to optimization_results/optimization_summary.json")
    print("2. The trading_algorithm.py file will automatically load these parameters")
    print("3. For interactive tuning, run: python run_optimization.py --dashboard")

if __name__ == "__main__":
    main()