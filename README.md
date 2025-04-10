# IMC Prosperity Trading Algorithm

This repository contains my trading algorithm for the IMC Trading Prosperity Challenge with optimization framework.

## Key Files

- `trading_algorithm.py`: Main trading algorithm that uses optimized parameters
- `optimization_framework.py`: Framework for backtesting and optimizing strategies 
- `quick_optimization.py`: Streamlined optimization script for faster testing
- `parameter_tuning_dashboard.py`: Interactive dashboard for strategy visualization
- `run_optimization.py`: Script to run the optimization process

## Trading Strategies

The algorithm implements specialized trading strategies for each product:

1. **RAINFOREST_RESIN**: Mean reversion strategy with stable fair value (10000)
2. **KELP**: Mean reversion with dynamic moving average for products that fluctuate
3. **SQUID_INK**: Order book imbalance strategy for pattern-based trading

## Getting Started

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure the data files are in the correct directory structure:
   ```
   data/round-1-island-data-bottle/prices_round_1_day_*.csv
   data/round-1-island-data-bottle/trades_round_1_day_*.csv
   ```

### Running the Optimization

1. Run the optimization script:
   ```bash
   # For full optimization (takes longer but more thorough)
   python run_optimization.py
   
   # For quick optimization (faster but less comprehensive)
   python run_optimization.py --quick
   
   # To specify a different data folder
   python run_optimization.py --data-folder /path/to/data
   ```

2. Launch the parameter tuning dashboard:
   ```bash
   python run_optimization.py --dashboard
   # OR
   streamlit run parameter_tuning_dashboard.py
   ```

### Using the Trading Algorithm

After optimization, the best parameters are saved to `optimization_results/optimization_summary.json` and automatically loaded by the trading algorithm.

To submit to the IMC Prosperity platform:
```
Upload trading_algorithm.py and select the Trader class
```

## Features

- Automatic parameter optimization for each product
- Interactive visualization dashboard for parameter tuning
- Position limit management for all products
- Advanced order book analysis and imbalance detection
- Moving average calculation and trend detection
- Take-profit and stop-loss risk management
- Robust data preprocessing and feature engineering

## Optimization Results

After running the optimization, view the results in:

- `optimization_results/optimization_summary.csv`: Tabular format of best parameters
- `optimization_results/optimization_summary.txt`: Text report with details
- `optimization_results/optimization_summary.json`: Parameter file used by the algorithm
- `optimization_results/*.png`: Visualization charts for each strategy