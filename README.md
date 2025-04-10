# IMC Prosperity Trading with Bayesian Optimization

This repository contains an advanced trading algorithm for the IMC Prosperity Challenge with integrated Bayesian optimization for parameter tuning.

## Key Features

- Sophisticated trading strategies for each product:
  - RAINFOREST_RESIN: Mean reversion with stable fair value (10000)
  - KELP: Mean reversion with dynamic moving average for fluctuating products
  - SQUID_INK: Order book imbalance strategy for pattern-based trading
- Bayesian optimization for optimal parameter selection
- Robust error handling and parameter consistency
- Multiple optimization frameworks for different needs

## Quick Start

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   For Bayesian optimization, you'll also need:
   ```
   pip install scipy scikit-learn
   ```

### Running the Optimization

There are three ways to run the optimization, in order of increasing effectiveness:

1. **Quick Optimization** (fastest but least thorough):

   ```bash
   python run_optimization.py --quick
   ```

2. **Full Grid Search** (thorough but time-consuming):

   ```bash
   python run_optimization.py
   ```

3. **Bayesian Optimization** (most effective, recommended):
   ```bash
   python run_optimization.py --bayesian
   ```

You can also specify the number of iterations for the optimization:

```bash
python run_optimization.py --bayesian --iterations 50
```

To optimize just a specific product:

```bash
python run_optimization.py --bayesian --product RAINFOREST_RESIN
```

### Interactive Parameter Tuning

Launch the parameter tuning dashboard:

```bash
python run_optimization.py --dashboard
```

### Submitting to IMC Prosperity

After optimization, the best parameters are automatically saved to `optimization_results/optimization_summary.json` and will be loaded by the trading algorithm.

To submit to the platform, upload `trading_algorithm.py`.

## Optimization Framework

The system includes three optimization approaches:

1. **Quick Optimization** (`quick_optimization.py`): Fast, simplified grid search
2. **Full Grid Search** (`optimization_framework.py`): Comprehensive grid search
3. **Bayesian Optimization** (`bayesian_optimization.py`): Advanced optimization using Gaussian Processes

### How Bayesian Optimization Works

Unlike grid search, Bayesian optimization uses Gaussian Processes to model the performance landscape. It strategically samples parameter combinations by:

1. Building a probabilistic model of the objective function
2. Using an acquisition function (Expected Improvement) to determine where to sample next
3. Balancing exploration of unknown regions with exploitation of promising areas
4. Converging to optimal parameters more efficiently than grid search

## Trading Algorithm

The trading algorithm (`trading_algorithm.py`) implements sophisticated strategies:

- **Mean Reversion**: For RAINFOREST_RESIN and KELP
  - Identifies when prices deviate from moving averages
  - Uses dynamic thresholds based on volatility
  - Incorporates advanced inventory management
- **Order Book Imbalance**: For SQUID_INK
  - Analyzes order book depth and imbalance
  - Combines with trend indicators
  - Uses take-profit and stop-loss mechanisms

## Data Structure

The system expects data in the following format:

```
data/round-1-island-data-bottle/prices_round_1_day_{-2,-1,0}.csv
data/round-1-island-data-bottle/trades_round_1_day_{-2,-1,0}.csv
```

## Results and Visualization

After optimization, results are saved to:

- `optimization_results/optimization_summary.json`: Parameters used by the algorithm
- `optimization_results/bayesian_optimization_summary.csv`: Tabular results
- `optimization_results/bayesian_optimization_summary.txt`: Text report
- `optimization_results/*.png`: Performance visualizations

## Troubleshooting

- **Parameter format issues**: Make sure optimization results are in the correct format
- **Missing dependencies**: For Bayesian optimization, install sklearn and scipy
- **Data not found**: Verify data files are in the expected locations
- **Error handling**: The algorithm has built-in error handling for robustness

## Advanced Usage

- Modify parameter ranges in each optimization script
- Adjust acquisition function parameters for Bayesian optimization
- Customize trading strategy parameters directly in the algorithm

## Contributing

Feel free to enhance the algorithm with additional strategies and optimizations. Key areas for improvement:

- Support for more products and strategies
- Advanced machine learning features
- Reinforcement learning approaches
