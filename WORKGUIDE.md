# IMC Prosperity Trading System - Working Guide

This document explains how the different components of the trading system work together and how to troubleshoot common issues.

## File Structure and Purpose

- **perfect_trader.py**: Final submission file with optimized strategies for each product
- **quick_optimization.py**: Fast parameter optimization focusing on recent market data
- **optimization_framework.py**: Comprehensive backtesting and visualization framework
- **parameter_tuning_dashboard.py**: Interactive Streamlit dashboard for strategy visualization

## How the Files Work Together

1. **Data Handling**: 
   - All files look for data in the `data/round-1-island-data-bottle/` directory
   - Data files are named `prices_round_1_day_{i}.csv` and `trades_round_1_day_{i}.csv`
   - The day indices are -2, -1, and 0

2. **Workflow**:
   - Run `optimization_framework.py` for comprehensive optimization across all days
   - Use `quick_optimization.py` for rapid testing on specific days
   - View and adjust parameters with `parameter_tuning_dashboard.py`
   - Submit `perfect_trader.py` with your optimized parameters

## Running the System

### Full Optimization

```bash
python optimization_framework.py
```
This performs a comprehensive grid search across all parameters and days. Results are saved to the `optimization_results` directory as CSV files and visualizations.

### Quick Optimization

```bash
python quick_optimization.py
```
This performs a faster grid search focusing on the most recent day (day 0). Results are displayed in the console and optimized for a quicker development cycle.

### Interactive Dashboard

```bash
streamlit run parameter_tuning_dashboard.py
```
This launches an interactive dashboard where you can:
- Load data
- Select products and strategies
- Adjust parameters
- Run backtests and visualize results

### Final Submission

The `perfect_trader.py` file is the one you should submit to the IMC Prosperity Challenge. It contains:
- All required classes for the IMC interface
- Optimized parameters for each product
- Advanced trading strategies

## Troubleshooting

### Data Loading Issues

If files aren't being found:
- Ensure data files are in `data/round-1-island-data-bottle/`
- Check the file naming format (`prices_round_1_day_-2.csv`, etc.)
- Verify file permissions are correct

### Zero PnL Results

If optimization shows zero PnL:
- Check that you're analyzing the right day (`day_index=0` for most recent)
- Verify parameters are appropriate for the product
- Ensure position limits are correctly set
- Check for proper trade execution in the backtesting logic

### Strategy Compatibility

- All strategies work with IMC's `TradingState` format
- The dashboard can work with either optimizer (quick or full)
- Parameters are transferable between all components

## Performance Tips

1. **For fastest iteration**:
   - Use `quick_optimization.py` with a small parameter grid
   - Focus on day 0 (most recent) for initial testing

2. **For most comprehensive results**:
   - Use `optimization_framework.py` with full parameter grid
   - Run overnight if needed for complete analysis

3. **For visualization**:
   - Use the dashboard with pre-optimized parameters
   - Compare strategies and parameter combinations visually

4. **For final submission**:
   - Update `perfect_trader.py` with the best parameters
   - Test thoroughly before submission