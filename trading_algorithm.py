#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List
from dataclasses import dataclass
import json
import jsonpickle

# Import data models
class Listing:
    def __init__(self, symbol: str, product: str, denomination: str):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    def __init__(self, symbol: str, price: int, quantity: int, buyer: str = None, seller: str = None, timestamp: int = 0) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp


class Order:
    def __init__(self, symbol: str, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity


@dataclass
class ConversionObservation:
    bidPrice: float
    askPrice: float
    transportFees: float
    exportTariff: float
    importTariff: float
    sugarPrice: float
    sunlightIndex: float


class Observation:
    def __init__(self, plainValueObservations: Dict[str, int], conversionObservations: Dict[str, ConversionObservation]) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations


class TradingState:
    def __init__(self,
                 traderData: str,
                 timestamp: int,
                 listings: Dict[str, Listing],
                 order_depths: Dict[str, OrderDepth],
                 own_trades: Dict[str, List[Trade]],
                 market_trades: Dict[str, List[Trade]],
                 position: Dict[str, int],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations


class Trader:
    def __init__(self):
        """
        Initialize the trader with optimized parameters and required data structures
        """
        # Load optimized parameters if available, otherwise use defaults
        try:
            with open("optimization_results/optimization_summary.json", "r") as f:
                self.optimized_params = json.load(f)
        except:
            self.optimized_params = None
        
        # Position limits for each product as specified in the challenge
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }
        
        # Initialize state tracking variables
        self.product_state = {}
        self.price_history = {}
        self.last_mid_price = {}
        self.moving_averages = {}
        self.entry_prices = {}
        
        # Strategy parameters - will be overridden if optimized parameters are available
        # Shared parameters across all products
        self.shared_params = {
            "take_profit_threshold": 0.4,  # Profit target as percentage
            "max_history_length": 90       # Maximum length of price history to keep
        }
        
        # Product-specific parameters
        self.product_params = {
            "RAINFOREST_RESIN": {
                "strategy": "mean_reversion",
                # Basic strategy parameters
                "window_size": 30,
                "entry_threshold": 1.0,
                "exit_threshold": 0.7,
                "base_order_qty": 25,  # Base quantity for orders
                "position_limit": 50,  # Maximum allowed
                
                # Advanced parameters
                "fair_value_anchor": 10000.0,
                "anchor_blend_alpha": 0.08,  # Blend between anchor and market price
                "min_spread": 7,            # Minimum spread to place orders
                "volatility_spread_factor": 0.32,  # Adjusts spread based on volatility
                "inventory_skew_factor": 0.01,     # How much to skew prices based on inventory
                "reversion_threshold": 2           # How far from mean to trigger reversion
            },
            "KELP": {
                "strategy": "mean_reversion",
                # Basic strategy parameters
                "window_size": 50,
                "entry_threshold": 1.0,
                "exit_threshold": 0.7,
                "base_order_qty": 28,  # Base quantity for orders
                "position_limit": 50,  # Maximum allowed
                
                # Advanced parameters
                "ema_alpha": 0.05,               # EMA smoothing factor
                "min_spread": 2,                 # Minimum spread to place orders
                "volatility_spread_factor": 1.2, # Adjusts spread based on volatility
                "inventory_skew_factor": 0.015,  # How much to skew prices based on inventory
                "min_volatility_qty_factor": 1.1, # Reduces quantity when volatility is high
                "max_volatility_for_qty_reduction": 4.0,
                "imbalance_depth": 5,            # Depth to analyze order book imbalance
                "imbalance_fv_adjustment_factor": 0.36  # Adjusts fair value based on imbalance
            },
            "SQUID_INK": {
                "strategy": "order_book_imbalance",
                # Basic strategy parameters
                "imbalance_threshold": 0.2,
                "take_profit": 3.0,
                "stop_loss": 2.0,
                "base_order_qty": 20,  # Base quantity for orders
                "position_limit": 50,  # Maximum allowed
                
                # Advanced parameters
                "ema_alpha": 0.1,               # Smoothing factor for EMA calculation
                "trend_strength_threshold": 0.6, # How strong a trend must be to act on it
                "min_spread": 3,                # Minimum spread to place orders
                "volatility_spread_factor": 0.8, # Adjusts spread based on volatility
                "inventory_skew_factor": 0.02,   # How much to skew prices based on inventory
                "imbalance_depth": 3,            # Depth to analyze order book imbalance
                "reversal_threshold": 1.5        # Threshold for trend reversal signals
            }
        }
        
        # Override with optimized parameters if available
        self._load_optimized_params()

    def _load_optimized_params(self):
        """
        Load optimized parameters from optimization results
        """
        if not self.optimized_params:
            return
        
        # Load shared parameters if available
        if "shared" in self.optimized_params:
            for param, value in self.optimized_params["shared"].items():
                if param in self.shared_params:
                    self.shared_params[param] = value
            
        # Load parameters for each product
        for product, params in self.optimized_params.items():
            if product == "shared":
                continue  # Already handled above
                
            if product in self.product_params:
                strategy = params.get("strategy")
                if strategy:
                    self.product_params[product]["strategy"] = strategy
                
                # Load all available parameters for the product
                for param, value in params.items():
                    if param != "strategy" and param in self.product_params[product]:
                        self.product_params[product][param] = value

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Primary method called by the exchange. Processes market data and generates orders.
        
        Args:
            state: TradingState object containing all market information
            
        Returns:
            Dict mapping product symbols to lists of Order objects
            conversion count
            trader state data as string
        """
        # Process and update state data with new information
        self._process_state_data(state)
        
        # Initialize result dictionary to store orders for each product
        result = {}
        
        # Process each available product
        for product in state.listings:
            # Skip products without order depth information
            if product not in state.order_depths:
                result[product] = []
                continue
                
            # Current position in this product
            position = state.position.get(product, 0)
            
            # Position limit for this product
            position_limit = self.position_limits.get(product, 50)
            
            # Order depth contains all buy/sell orders from other participants
            order_depth = state.order_depths[product]
            
            # Get the strategy for this product
            strategy = self.product_params.get(product, {}).get("strategy", "default")
            
            # Select and execute the appropriate strategy for this product
            if strategy == "mean_reversion":
                orders = self._mean_reversion_strategy(product, order_depth, position, position_limit, state)
            elif strategy == "order_book_imbalance":
                orders = self._order_book_imbalance_strategy(product, order_depth, position, position_limit, state)
            else:
                # Default strategy for any new products
                orders = self._default_strategy(product, order_depth, position, position_limit)
                
            result[product] = orders
            
        # Serialize state data for the next iteration
        trader_data = self._serialize_state()
        
        # No conversions by default
        conversions = 0
            
        return result, conversions, trader_data
        
    def _process_state_data(self, state: TradingState):
        """
        Process incoming state data and update internal tracking
        """
        # Initialize state for new products
        for product in state.listings:
            if product not in self.price_history:
                self.price_history[product] = []
                self.moving_averages[product] = []
                
            # Process order book and compute mid price if order book exists
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                
                # Calculate mid price if there are both buy and sell orders
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                    spread = best_ask - best_bid
                    
                    # Store mid price and timestamp
                    self.price_history[product].append((state.timestamp, mid_price))
                    
                    # Limit history length based on shared parameter
                    max_history = self.shared_params.get("max_history_length", 90)
                    if len(self.price_history[product]) > max_history:
                        self.price_history[product] = self.price_history[product][-max_history:]
                    
                    self.last_mid_price[product] = mid_price
                    
                    # Initialize product state if not exists
                    if product not in self.product_state:
                        self.product_state[product] = {}
                    
                    # Store spread information
                    self.product_state[product]['spread'] = spread
                    
                    # Calculate order book imbalance at multiple depths
                    # Basic imbalance (all levels)
                    total_bid_volume = sum(abs(qty) for qty in order_depth.buy_orders.values())
                    total_ask_volume = sum(abs(qty) for qty in order_depth.sell_orders.values())
                    
                    if total_bid_volume + total_ask_volume > 0:
                        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
                    else:
                        imbalance = 0
                    
                    self.product_state[product]['imbalance'] = imbalance
                    
                    # Calculate advanced order book metrics
                    imbalance_depth = self.product_params.get(product, {}).get('imbalance_depth', 3)
                    
                    # Imbalance at specified depth (if enough levels exist)
                    bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
                    ask_prices = sorted(order_depth.sell_orders.keys())
                    
                    # Limited depth imbalance
                    depth_bid_volume = sum(abs(order_depth.buy_orders[p]) for p in bid_prices[:imbalance_depth] if p in order_depth.buy_orders)
                    depth_ask_volume = sum(abs(order_depth.sell_orders[p]) for p in ask_prices[:imbalance_depth] if p in order_depth.sell_orders)
                    
                    if depth_bid_volume + depth_ask_volume > 0:
                        depth_imbalance = (depth_bid_volume - depth_ask_volume) / (depth_bid_volume + depth_ask_volume)
                    else:
                        depth_imbalance = 0
                        
                    self.product_state[product]['depth_imbalance'] = depth_imbalance
                    
                    # Calculate short-term volatility (based on recent price movements)
                    if len(self.price_history[product]) >= 5:
                        recent_prices = [price for _, price in self.price_history[product][-5:]]
                        if len(recent_prices) > 1:
                            # Calculate percent changes
                            pct_changes = [abs((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]) 
                                          for i in range(1, len(recent_prices))]
                            volatility = sum(pct_changes) / len(pct_changes) * 100  # as percentage
                            self.product_state[product]['volatility'] = volatility
                    
        # Process own trades to track entry prices
        for product, trades in state.own_trades.items():
            for trade in trades:
                if product not in self.entry_prices:
                    self.entry_prices[product] = {}
                    
                # If this is our first position or a position reversal, record entry price
                if (product in state.position and 
                    ((trade.buyer == "SUBMISSION" and state.position[product] > 0) or 
                     (trade.seller == "SUBMISSION" and state.position[product] < 0))):
                    self.entry_prices[product]['price'] = trade.price
                    self.entry_prices[product]['timestamp'] = state.timestamp
                    
        # Update moving averages for each product
        self._update_moving_averages(state)
        
    def _update_moving_averages(self, state: TradingState):
        """
        Update moving averages for all products
        """
        for product in self.price_history:
            if product in self.product_params:
                params = self.product_params[product]
                window_size = params.get('window_size', 20)
                ema_alpha = params.get('ema_alpha', 0.1)  # Default EMA smoothing factor
                
                # Keep only recent price history
                max_history = self.shared_params.get('max_history_length', 90)
                if len(self.price_history[product]) > max_history:
                    self.price_history[product] = self.price_history[product][-max_history:]
                
                # Make sure we have product state initialized
                if product not in self.product_state:
                    self.product_state[product] = {}
                
                # Calculate simple moving average if we have enough data
                if len(self.price_history[product]) >= window_size:
                    recent_prices = [price for _, price in self.price_history[product][-window_size:]]
                    sma = sum(recent_prices) / len(recent_prices)
                    self.product_state[product]['sma'] = sma
                    
                    # Calculate price deviation from SMA
                    if product in self.last_mid_price and sma > 0:
                        deviation = (self.last_mid_price[product] - sma) / sma
                        self.product_state[product]['deviation'] = deviation
                        
                        # Store deviation in multiples of reversion threshold for mean reversion
                        reversion_threshold = params.get('reversion_threshold', 2.0)
                        if reversion_threshold > 0:
                            self.product_state[product]['deviation_score'] = deviation / reversion_threshold
                
                # Calculate exponential moving average (EMA)
                if product in self.last_mid_price:
                    current_price = self.last_mid_price[product]
                    
                    # Initialize EMA if not exists
                    if 'ema' not in self.product_state[product]:
                        self.product_state[product]['ema'] = current_price
                    
                    # Update EMA
                    prev_ema = self.product_state[product]['ema']
                    new_ema = (current_price * ema_alpha) + (prev_ema * (1 - ema_alpha))
                    self.product_state[product]['ema'] = new_ema
                    
                    # Calculate EMA-based signals
                    if prev_ema > 0:
                        # EMA trend direction and strength
                        ema_change = (new_ema - prev_ema) / prev_ema
                        self.product_state[product]['ema_trend'] = ema_change
                        
                # For RAINFOREST_RESIN, calculate fair value as blend of anchor and market price
                if product == 'RAINFOREST_RESIN' and product in self.last_mid_price:
                    anchor_value = params.get('fair_value_anchor', 10000.0)
                    blend_alpha = params.get('anchor_blend_alpha', 0.08)
                    
                    # Calculate fair value based on blend of anchor and EMA/SMA
                    if 'ema' in self.product_state[product]:
                        market_value = self.product_state[product]['ema']
                    elif 'sma' in self.product_state[product]:
                        market_value = self.product_state[product]['sma']
                    else:
                        market_value = self.last_mid_price[product]
                    
                    # Blend anchor value with market value
                    fair_value = (market_value * blend_alpha) + (anchor_value * (1 - blend_alpha))
                    self.product_state[product]['fair_value'] = fair_value
                    
    def _mean_reversion_strategy(self, product, order_depth, position, position_limit, state):
        """
        Advanced mean reversion strategy with inventory management and volatility adjustment
        """
        params = self.product_params[product]
        
        # Extract all parameters with defaults
        window_size = params.get('window_size', 30)
        entry_threshold = params.get('entry_threshold', 1.0)
        exit_threshold = params.get('exit_threshold', 0.7)
        base_order_qty = params.get('base_order_qty', 25)
        min_spread = params.get('min_spread', 2)
        inventory_skew_factor = params.get('inventory_skew_factor', 0.01)
        volatility_spread_factor = params.get('volatility_spread_factor', 0.3)
        take_profit_threshold = self.shared_params.get('take_profit_threshold', 0.4)
        
        # Get product state
        product_state = self.product_state.get(product, {})
        
        orders = []
        
        # Get best bid and ask if available
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Skip if spread is too narrow (not profitable enough)
        if spread < min_spread:
            return orders
        
        # Determine reference price (fair value)
        if product == 'RAINFOREST_RESIN' and 'fair_value' in product_state:
            reference_price = product_state['fair_value']
        elif 'ema' in product_state:
            reference_price = product_state['ema']
        elif 'sma' in product_state:
            reference_price = product_state['sma']
        else:
            # No reference price available yet
            return orders
            
        # Calculate deviation from reference
        deviation = (mid_price - reference_price) / reference_price
        
        # Get current volatility for dynamic adjustments
        volatility = product_state.get('volatility', 0)
        
        # Adjust thresholds based on volatility
        vol_adjusted_entry = entry_threshold * (1 + volatility * volatility_spread_factor / 100)
        vol_adjusted_exit = exit_threshold * (1 + volatility * volatility_spread_factor / 100)
        
        # Adjust order size based on volatility for KELP
        if product == 'KELP' and 'min_volatility_qty_factor' in params:
            min_vol_factor = params.get('min_volatility_qty_factor', 1.0)
            max_vol = params.get('max_volatility_for_qty_reduction', 4.0)
            
            # Reduce order size as volatility increases
            if volatility > 0:
                vol_ratio = min(volatility / max_vol, 1.0)
                size_factor = 1.0 - (1.0 - min_vol_factor) * vol_ratio
                adjusted_order_size = max(1, int(base_order_qty * size_factor))
            else:
                adjusted_order_size = base_order_qty
        else:
            adjusted_order_size = base_order_qty
            
        # Calculate inventory skew to avoid taking too much risk on one side
        inventory_ratio = position / position_limit if position_limit > 0 else 0
        skew_adjustment = inventory_ratio * inventory_skew_factor * reference_price
        
        # Adjust reference price based on inventory position
        # This makes it harder to add to position as it grows
        skewed_reference = reference_price - skew_adjustment
        
        # For KELP, adjust reference price based on order book imbalance
        if product == 'KELP' and 'imbalance_fv_adjustment_factor' in params:
            imbalance = product_state.get('depth_imbalance', 0)
            imbalance_factor = params.get('imbalance_fv_adjustment_factor', 0.2)
            
            # Adjust fair value upward for positive imbalance (more bids than asks)
            imbalance_adjustment = imbalance * imbalance_factor * reference_price
            skewed_reference += imbalance_adjustment
        
        # Calculate buy and sell prices
        buy_price = best_ask  # Default to market orders
        sell_price = best_bid
        
        # Check if we can place better limit orders inside the spread for tight spreads
        if spread > min_spread * 2:
            # Place limit orders inside the spread
            buy_price = int(best_bid + spread * 0.3)
            sell_price = int(best_ask - spread * 0.3)
        
        # Trading logic
        # 1. Take profit logic for existing positions
        if position != 0 and product in self.entry_prices and 'price' in self.entry_prices[product]:
            entry_price = self.entry_prices[product]['price']
            price_change_pct = (mid_price - entry_price) / entry_price
            
            # For long positions, check for profit target
            if position > 0 and price_change_pct >= take_profit_threshold:
                sell_quantity = min(position, order_depth.buy_orders[best_bid])
                if sell_quantity > 0:
                    orders.append(Order(product, best_bid, -sell_quantity))
                    return orders  # Early return after taking profit
                    
            # For short positions, check for profit target
            elif position < 0 and price_change_pct <= -take_profit_threshold:
                buy_quantity = min(-position, abs(order_depth.sell_orders[best_ask]))
                if buy_quantity > 0:
                    orders.append(Order(product, best_ask, buy_quantity))
                    return orders  # Early return after taking profit
        
        # 2. Mean reversion logic based on deviation from fair value
        if position == 0:
            # No position - look for entry opportunities
            
            # If price is significantly below reference, BUY
            if deviation < -vol_adjusted_entry and position < position_limit:
                # Dynamically calculate order size based on deviation strength
                signal_strength = min(3.0, abs(deviation) / vol_adjusted_entry)
                target_quantity = int(adjusted_order_size * signal_strength)
                
                # Limit by position limit
                quantity = min(target_quantity, position_limit - position)
                buy_quantity = min(quantity, abs(order_depth.sell_orders[best_ask]))
                if buy_quantity > 0:
                    orders.append(Order(product, buy_price, buy_quantity))
            
            # If price is significantly above reference, SELL
            elif deviation > vol_adjusted_entry and position > -position_limit:
                # Dynamically calculate order size based on deviation strength
                signal_strength = min(3.0, abs(deviation) / vol_adjusted_entry)
                target_quantity = int(adjusted_order_size * signal_strength)
                
                # Limit by position limit
                quantity = min(target_quantity, position_limit + position)
                sell_quantity = min(quantity, order_depth.buy_orders[best_bid])
                if sell_quantity > 0:
                    orders.append(Order(product, sell_price, -sell_quantity))
                    
        else:
            # Have a position - look for exit opportunities
            
            # If in a long position and price has reverted close to reference, SELL
            if position > 0 and abs(deviation) < vol_adjusted_exit:
                # Sell our position at the current best bid
                sell_quantity = min(position, order_depth.buy_orders[best_bid])
                if sell_quantity > 0:
                    orders.append(Order(product, sell_price, -sell_quantity))
            
            # If in a short position and price has reverted close to reference, BUY
            elif position < 0 and abs(deviation) < vol_adjusted_exit:
                # Buy to cover our short at the current best ask
                buy_quantity = min(-position, abs(order_depth.sell_orders[best_ask]))
                if buy_quantity > 0:
                    orders.append(Order(product, buy_price, buy_quantity))
        
        return orders
    
    def _order_book_imbalance_strategy(self, product, order_depth, position, position_limit, state):
        """
        Advanced order book imbalance strategy with trend detection and volatility adjustment
        """
        params = self.product_params[product]
        
        # Extract parameters with defaults
        imbalance_threshold = params.get('imbalance_threshold', 0.2)
        take_profit = params.get('take_profit', 3.0)
        stop_loss = params.get('stop_loss', 2.0)
        base_order_qty = params.get('base_order_qty', 20)
        min_spread = params.get('min_spread', 3)
        inventory_skew_factor = params.get('inventory_skew_factor', 0.02)
        trend_strength_threshold = params.get('trend_strength_threshold', 0.6)
        reversal_threshold = params.get('reversal_threshold', 1.5)
        volatility_spread_factor = params.get('volatility_spread_factor', 0.8)
        imbalance_depth = params.get('imbalance_depth', 3)
        
        orders = []
        
        # Get best bid and ask if available
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Skip if spread is too narrow (not profitable enough)
        if spread < min_spread:
            return orders
            
        # Get product state data
        product_state = self.product_state.get(product, {})
        
        # Get imbalance metrics
        imbalance = product_state.get('imbalance', 0)
        depth_imbalance = product_state.get('depth_imbalance', 0)  # Imbalance at specified depth
        
        # Use depth imbalance if available (more accurately reflects real trading pressure)
        if abs(depth_imbalance) > abs(imbalance):
            effective_imbalance = depth_imbalance
        else:
            effective_imbalance = imbalance
            
        # Get trend information from EMA
        ema_trend = product_state.get('ema_trend', 0)
        
        # Get volatility information
        volatility = product_state.get('volatility', 0)
        
        # Adjust thresholds based on volatility
        vol_adjusted_imbalance = imbalance_threshold * (1 + volatility * volatility_spread_factor / 100)
        
        # Adjust order size based on imbalance strength
        imbalance_strength = min(3.0, abs(effective_imbalance) / vol_adjusted_imbalance)
        adjusted_order_size = max(1, int(base_order_qty * imbalance_strength))
        
        # Calculate inventory skew to reduce risk of overexposure
        inventory_ratio = position / position_limit if position_limit > 0 else 0
        inventory_factor = 1.0 - abs(inventory_ratio) * inventory_skew_factor
        
        # Further adjust order size based on inventory
        inventory_adjusted_size = max(1, int(adjusted_order_size * inventory_factor))
        
        # Determine market direction
        # Combine imbalance signal with trend signal for stronger entry conditions
        imbalance_signal = 1 if effective_imbalance > vol_adjusted_imbalance else (-1 if effective_imbalance < -vol_adjusted_imbalance else 0)
        trend_signal = 1 if ema_trend > trend_strength_threshold else (-1 if ema_trend < -trend_strength_threshold else 0)
        
        # Combined signal - stronger when both trend and imbalance agree
        combined_signal = 0
        if imbalance_signal != 0 and trend_signal != 0:
            # Both signals agree - strong signal
            if imbalance_signal == trend_signal:
                combined_signal = imbalance_signal * 2
            # Signals conflict - weak or no signal
            else:
                combined_signal = 0
        else:
            # Only one signal is non-zero
            combined_signal = imbalance_signal + trend_signal
            
        # Trading logic based on position
        if position == 0:
            # No position - look for entry opportunities based on combined signals
            
            # If strong buy signal, go long
            if combined_signal > 0 and position < position_limit:
                # Buy at the current best ask price
                quantity = min(inventory_adjusted_size, position_limit - position)
                buy_quantity = min(quantity, abs(order_depth.sell_orders[best_ask]))
                if buy_quantity > 0:
                    orders.append(Order(product, best_ask, buy_quantity))
            
            # If strong sell signal, go short
            elif combined_signal < 0 and position > -position_limit:
                # Sell at the current best bid price
                quantity = min(inventory_adjusted_size, position_limit + position)
                sell_quantity = min(quantity, order_depth.buy_orders[best_bid])
                if sell_quantity > 0:
                    orders.append(Order(product, best_bid, -sell_quantity))
                    
        else:
            # Have a position - look for exit opportunities
            
            # Check if we have entry price information
            if product in self.entry_prices and 'price' in self.entry_prices[product]:
                entry_price = self.entry_prices[product]['price']
                entry_time = self.entry_prices[product].get('timestamp', 0)
                current_time = state.timestamp
                
                # Calculate price change as percentage
                price_change_pct = (mid_price - entry_price) / entry_price * 100
                
                # Calculate position duration
                position_duration = current_time - entry_time
                
                if position > 0:  # Long position
                    # Take profit or stop loss for long position
                    if price_change_pct >= take_profit or price_change_pct <= -stop_loss:
                        # Sell our position at the current best bid
                        sell_quantity = min(position, order_depth.buy_orders[best_bid])
                        if sell_quantity > 0:
                            orders.append(Order(product, best_bid, -sell_quantity))
                            return orders  # Early return after taking profit/stopping loss
                
                elif position < 0:  # Short position
                    # Take profit or stop loss for short position
                    if price_change_pct <= -take_profit or price_change_pct >= stop_loss:
                        # Buy to cover our short at the current best ask
                        buy_quantity = min(-position, abs(order_depth.sell_orders[best_ask]))
                        if buy_quantity > 0:
                            orders.append(Order(product, best_ask, buy_quantity))
                            return orders  # Early return after taking profit/stopping loss
            
            # Exit on strong imbalance reversal
            if (position > 0 and effective_imbalance < -vol_adjusted_imbalance * reversal_threshold) or \
               (position < 0 and effective_imbalance > vol_adjusted_imbalance * reversal_threshold):
                
                if position > 0:
                    # Sell our position at the current best bid
                    sell_quantity = min(position, order_depth.buy_orders[best_bid])
                    if sell_quantity > 0:
                        orders.append(Order(product, best_bid, -sell_quantity))
                elif position < 0:
                    # Buy to cover our short at the current best ask
                    buy_quantity = min(-position, abs(order_depth.sell_orders[best_ask]))
                    if buy_quantity > 0:
                        orders.append(Order(product, best_ask, buy_quantity))
        
        return orders
    
    def _default_strategy(self, product, order_depth, position, position_limit):
        """
        Default strategy for any new products
        """
        orders = []
        
        # Get best bid and ask if available
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # Simple market making strategy - place orders inside the spread if it's wide enough
        spread = best_ask - best_bid
        
        # Only place orders if the spread is wide enough to make a profit
        if spread > 2:
            # Place a buy order just above the best bid
            buy_price = best_bid + 1
            buy_volume = min(5, position_limit - position)
            
            # Place a sell order just below the best ask
            sell_price = best_ask - 1
            sell_volume = min(5, position_limit + position)
            
            # Add orders if volumes are positive
            if buy_volume > 0:
                orders.append(Order(product, buy_price, buy_volume))
            if sell_volume > 0:
                orders.append(Order(product, sell_price, -sell_volume))
        
        return orders
    
    def _serialize_state(self):
        """
        Convert internal state to string for persistence between iterations
        """
        # For simplicity, just return an empty string
        # In a real implementation, you'd use jsonpickle to serialize important state
        return ""