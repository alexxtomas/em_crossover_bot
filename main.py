import time
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta

class EMA_Crossover_Bot:
    def __init__(self, symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, ema_fast=80, ema_slow=280, risk_percent=1.0):
        """Initialize the EMA Crossover Bot"""
        self.symbol = symbol
        self.timeframe = timeframe
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.risk_percent = risk_percent
        self.position = None
        self.initialized = False
        
    def initialize(self):
        """Connect to MetaTrader 5"""
        if not mt5.initialize():
            print(f"initialize() failed, error code = {mt5.last_error()}")
            return False
        
        # Check if the symbol exists
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Symbol {self.symbol} not found. Please check the symbol name.")
            mt5.shutdown()
            return False
            
        # Make sure the symbol is visible and trades are allowed
        if not symbol_info.visible:
            print(f"Symbol {self.symbol} is not visible, trying to switch on")
            if not mt5.symbol_select(self.symbol, True):
                print(f"symbol_select({self.symbol}) failed, error code = {mt5.last_error()}")
                mt5.shutdown()
                return False
                
        print(f"Connected to MetaTrader 5, using symbol {self.symbol}")
        self.initialized = True
        return True
        
    def fetch_data(self):
        """Fetch historical data for the symbol"""
        if not self.initialized:
            print("Bot not initialized. Please call initialize() first.")
            return None
            
        # Define the time range to retrieve
        current_time = datetime.now()
        time_from = current_time - timedelta(days=7)  # Get data for last 7 days
        
        # Fetch the bars
        bars = mt5.copy_rates_from(self.symbol, self.timeframe, time_from, 1000)
        if bars is None or len(bars) == 0:
            print(f"Failed to retrieve bars, error code = {mt5.last_error()}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(bars)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calculate EMAs
        df[f'ema{self.ema_fast}'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df[f'ema{self.ema_slow}'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        return df
        
    def calculate_signals(self, df):
        """Calculate trading signals based on EMA crossover"""
        # Calculate crossover
        df['signal'] = 0
        
        # EMA fast crosses above EMA slow (bullish)
        df.loc[(df[f'ema{self.ema_fast}'] > df[f'ema{self.ema_slow}']) & 
               (df[f'ema{self.ema_fast}'].shift(1) <= df[f'ema{self.ema_slow}'].shift(1)), 'signal'] = 1
        
        # EMA fast crosses below EMA slow (bearish)
        df.loc[(df[f'ema{self.ema_fast}'] < df[f'ema{self.ema_slow}']) & 
               (df[f'ema{self.ema_fast}'].shift(1) >= df[f'ema{self.ema_slow}'].shift(1)), 'signal'] = -1
        
        return df
        
    def calculate_position_size(self, price, stop_loss, account_balance=None):
        """Calculate position size based on risk percentage"""
        if account_balance is None:
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account info")
                return 0.01  # Default minimal size
            account_balance = account_info.balance
            
        # For XAU/USD, position size is typically in troy ounces
        # Risk amount in account currency
        risk_amount = account_balance * (self.risk_percent / 100)
        
        # Calculate pip value
        point_value = mt5.symbol_info(self.symbol).point
        pip_value = point_value * 10  # Typically a pip is 10 points for XAUUSD
        
        # Calculate distance to stop loss in pips
        sl_distance_in_pips = abs(price - stop_loss) / pip_value
        
        # Calculate position size
        contract_size = mt5.symbol_info(self.symbol).trade_contract_size  # Typically 100 for XAUUSD
        
        # Calculate pip value in account currency
        if sl_distance_in_pips == 0:
            return 0.01  # Minimum position size
            
        position_size = risk_amount / (sl_distance_in_pips * pip_value * contract_size)
        
        # Round down to 2 decimal places (typical for XAUUSD)
        position_size = max(round(position_size, 2), 0.01)
        
        return position_size
        
    def place_order(self, order_type, price, sl=None):
        """Place an order using MT5"""
        if not self.initialized:
            print("Bot not initialized. Please call initialize() first.")
            return False
            
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Failed to get symbol info for {self.symbol}")
            return False
            
        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": 0.01,  # Default minimal value, will be updated
            "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 10,  # Allow price deviation in points
            "magic": 123456,  # Magic number for identifying bot orders
            "comment": "EMA Crossover Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Calculate stop loss price
        if sl is not None:
            request["sl"] = sl
            # Calculate appropriate position size based on risk
            request["volume"] = self.calculate_position_size(price, sl)
            
        # Send the order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed, retcode={result.retcode}")
            print(f"Error description: {result.comment}")
            return False
            
        print(f"Order placed successfully: {order_type} {request['volume']} {self.symbol} at {price}")
        return True
        
    def check_existing_positions(self):
        """Check for existing positions on the symbol"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            print("No positions found")
            return None
            
        if len(positions) > 0:
            for position in positions:
                if position.magic == 123456:  # Check if it's our bot's position
                    return position.type  # 0 for buy, 1 for sell
        return None
        
    def close_position(self):
        """Close any existing position for the symbol"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None or len(positions) == 0:
            return True  # No positions to close
            
        for position in positions:
            if position.magic == 123456:  # Only close our bot's positions
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": mt5.symbol_info_tick(self.symbol).bid if position.type == 0 else mt5.symbol_info_tick(self.symbol).ask,
                    "deviation": 10,
                    "magic": 123456,
                    "comment": "EMA Crossover Bot - Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Failed to close position, retcode={result.retcode}")
                    return False
                print(f"Position closed successfully: ticket={position.ticket}")
        return True
        
    def run(self, check_interval=60):
        """Run the trading bot in a loop"""
        if not self.initialize():
            return
            
        print(f"Starting EMA Crossover Bot for {self.symbol}")
        print(f"Strategy: EMA{self.ema_fast} crosses EMA{self.ema_slow}")
        print(f"Risk: {self.risk_percent}% per trade")
        
        while True:
            try:
                # Fetch current data
                data = self.fetch_data()
                if data is None or len(data) < self.ema_slow:
                    print("Not enough data to calculate EMAs")
                    time.sleep(check_interval)
                    continue
                    
                # Calculate signals
                data = self.calculate_signals(data)
                
                # Get latest signal
                latest_signal = data.iloc[-1]['signal']
                
                # Check current positions
                current_position = self.check_existing_positions()
                
                # Get current price
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is None:
                    print("Failed to get current price")
                    time.sleep(check_interval)
                    continue
                
                bid, ask = tick.bid, tick.ask
                
                # Trading logic
                if latest_signal == 1:  # Bullish signal
                    if current_position == 1:  # Already have a sell position
                        print("Closing SELL position due to bullish crossover")
                        self.close_position()
                        
                    if current_position != 0:  # No buy position exists
                        print("EMA80 crossed above EMA280 - BUY Signal")
                        # Calculate stop loss - simple example, you may want a more sophisticated approach
                        stop_loss = bid * 0.99  # 1% below current price
                        self.place_order("BUY", ask, stop_loss)
                        
                elif latest_signal == -1:  # Bearish signal
                    if current_position == 0:  # Already have a buy position
                        print("Closing BUY position due to bearish crossover")
                        self.close_position()
                        
                    if current_position != 1:  # No sell position exists
                        print("EMA80 crossed below EMA280 - SELL Signal")
                        # Calculate stop loss - simple example
                        stop_loss = ask * 1.01  # 1% above current price
                        self.place_order("SELL", bid, stop_loss)
                
                print(f"Last check: {datetime.now()}, EMA80: {data.iloc[-1][f'ema{self.ema_fast}']:.2f}, "
                      f"EMA280: {data.iloc[-1][f'ema{self.ema_slow}']:.2f}")
                      
            except Exception as e:
                print(f"Error in main loop: {e}")
                
            # Wait for next check
            time.sleep(check_interval)
            
    def shutdown(self):
        """Properly shutdown MT5 connection"""
        mt5.shutdown()
        print("MetaTrader 5 connection closed")
        
if __name__ == "__main__":
    # Create and run the bot
    bot = EMA_Crossover_Bot(
        symbol="XAUUSD",
        timeframe=mt5.TIMEFRAME_M5,
        ema_fast=80,
        ema_slow=280,
        risk_percent=1.0
    )
    
    try:
        bot.run(check_interval=60)  # Check every minute
    except KeyboardInterrupt:
        print("Bot stopped by user")
    finally:
        bot.shutdown()