# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

- Run optimization: `python optimization_framework.py`
- Run dashboard: `streamlit run parameter_tuning_dashboard.py`
  To get started:
  1. Run `python run_optimization.py --quick` for a quick test
  2. Use `python run_optimization.py --dashboard` to
     interactively tune parameters
  3. For full optimization, run `python run_optimization.py`

## Code Style Guidelines

- Imports: Use standard library, then third-party, then local imports
- Formatting: PEP 8 compliant, with docstrings
- Types: Use type hints (from typing import Dict, List, Tuple)
- Naming: snake_case for variables/functions, CamelCase for classes
- Error handling: Use try/except with specific exception types
- Documentation: Use docstrings for functions and classes
- Data handling: Prefer pandas for data manipulation
- Visualization: Use matplotlib/seaborn with consistent style
- Parameters: Keep naming consistent across strategies

## Project Structure

- `optimization_framework.py`: Main optimization classes and strategies
- `parameter_tuning_dashboard.py`: Streamlit dashboard for tuning
- `data/`: Directory for price and trade data
