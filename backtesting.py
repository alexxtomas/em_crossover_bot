import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import MetaTrader5 as mt5  # For historical data fetching
import pytz
from matplotlib.dates import DateFormatter
import os

class EMA_Crossover_Backtest:
    def __init__(self, symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, ema_fast=80, ema_slow=200, 
                 risk_percent=1.0, initial_capital=10000, start_date=None, end_date=None):
        """Initialize the EMA Crossover Backtester"""
        self.symbol = symbol
        self.timeframe = timeframe
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.risk_percent = risk_percent
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Default to last 6 months if dates not provided
        if start_date is None:
            self.start_date = datetime.now() - timedelta(days=180)
        else:
            self.start_date = start_date
            
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = end_date
            
        # For tracking trades and performance
        self.trades = []
        self.equity_curve = []
        
        # Connect to MT5
        self.mt5_connected = False
    
    def connect_to_mt5(self):
        """Connect to MetaTrader 5 to fetch historical data"""
        # VERY IMPORTANT: Replace with YOUR ACTUAL PATH
        mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
        print(f"Attempting to connect to MT5 at path: {mt5_path}")
        if not os.path.exists(mt5_path):
            print(f"ERROR: MT5 executable not found at: {mt5_path}")
            print("       Please check the path and update mt5_path.")
            return False
        if not mt5.initialize(path=mt5_path, timeout=3000, login=5034026957, password="*0NzOcSs", server="MetaQuotes-Demo"):
            print(f"initialize() failed, error code = {mt5.last_error()}")
            print("       Check MT5 is running, automated trading is enabled,")
            print("       and DLL imports are allowed in MT5 settings.")
            return False
        # Check if the symbol exists
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Symbol {self.symbol} not found. Please check the symbol name.")
            mt5.shutdown()
            return False
        # Make sure the symbol is visible
        if not symbol_info.visible:
            print(f"Symbol {self.symbol} is not visible, trying to switch on")
            if not mt5.symbol_select(self.symbol, True):
                print(f"symbol_select({self.symbol}) failed, error code = {mt5.last_error()}")
                mt5.shutdown()
                return False
        print(f"Connected to MetaTrader 5, using symbol {self.symbol}")
        self.mt5_connected = True
        return True
        
    def fetch_historical_data(self):
        """Fetch historical price data from MT5 in smaller chunks"""
        if not self.mt5_connected:
            if not self.connect_to_mt5():
                print("Failed to connect to MT5. Unable to fetch historical data.")
                return None
        # Convert datetime to UTC for MT5
        timezone = pytz.timezone("Etc/UTC")
        end_date_utc = timezone.localize(self.end_date)
        start_date_utc = timezone.localize(self.start_date)
        
        if self.timeframe == mt5.TIMEFRAME_M5:
            print(f"Fetching M5 data in chunks for symbol: {self.symbol}")
            all_bars = []
            current_start = start_date_utc
            chunk_size = timedelta(days=14)
            while current_start < end_date_utc:
                current_end = min(current_start + chunk_size, end_date_utc)
                print(f"Fetching chunk: {current_start} to {current_end}")
                chunk_bars = mt5.copy_rates_range(self.symbol, self.timeframe, current_start, current_end)
                if chunk_bars is not None and len(chunk_bars) > 0:
                    all_bars.extend(chunk_bars)
                    print(f"Fetched {len(chunk_bars)} bars for this chunk")
                else:
                    print(f"Failed to fetch data for chunk: {mt5.last_error()}")
                current_start = current_end
            if not all_bars:
                print("No data was fetched in any chunk")
                return None
            bars = np.array(all_bars)
        else:
            print(f"Attempting to fetch data for symbol: {self.symbol}, timeframe: {self.timeframe}")
            print(f"Date range: {start_date_utc} to {end_date_utc}")
            bars = mt5.copy_rates_range(self.symbol, self.timeframe, start_date_utc, end_date_utc)
        if bars is None or len(bars) == 0:
            print(f"Failed to fetch historical data: {mt5.last_error()}")
            return None
        df = pd.DataFrame(bars)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df[f'ema{self.ema_fast}'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df[f'ema{self.ema_slow}'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        df = df.iloc[self.ema_slow:]
        print(f"Fetched {len(df)} bars from {df['time'].min()} to {df['time'].max()}")
        return df
        
    def calculate_signals(self, df):
        """Calculate trading signals based on EMA crossover"""
        df['signal'] = 0
        df.loc[(df[f'ema{self.ema_fast}'] > df[f'ema{self.ema_slow}']) & 
               (df[f'ema{self.ema_fast}'].shift(1) <= df[f'ema{self.ema_slow}'].shift(1)), 'signal'] = 1
        df.loc[(df[f'ema{self.ema_fast}'] < df[f'ema{self.ema_slow}']) & 
               (df[f'ema{self.ema_fast}'].shift(1) >= df[f'ema{self.ema_slow}'].shift(1)), 'signal'] = -1
        df['position'] = 0
        position = 0
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1:
                position = 1
            elif df['signal'].iloc[i] == -1:
                position = -1
            df['position'].iloc[i] = position
        return df
        
    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk percentage"""
        risk_amount = self.capital * (self.risk_percent / 100)
        price_difference = abs(entry_price - stop_loss)
        risk_per_lot = price_difference * 100
        if risk_per_lot == 0:
            return 0.01
        position_size = risk_amount / risk_per_lot
        position_size = max(round(position_size, 2), 0.01)
        return position_size
        
    def backtest(self):
        """Run the backtest with delayed entries and trailing stops to reduce drawdowns"""
        data = self.fetch_historical_data()
        if data is None:
            print("No data available for backtesting")
            return None, None
        data = self.calculate_signals(data)
        self.capital = self.initial_capital
        self.equity_curve = [self.capital]
        current_position = 0
        position_size = 0
        entry_price = 0
        entry_time = None
        stop_loss = 0
        
        delay_bars = 3  # Delay before confirming entry
        trailing_stop_pct = 0.005  # 0.5% trailing stop
        i = 1
        
        while i < len(data):
            current_time = data['time'].iloc[i]
            current_price = data['close'].iloc[i]
            current_signal = data['signal'].iloc[i]
            self.equity_curve.append(self.capital)
            
            # If no open position, check for a signal and wait for confirmation delay
            if current_position == 0:
                if current_signal in [1, -1]:
                    pending_signal = current_signal
                    if i + delay_bars < len(data):
                        confirmed = True
                        for j in range(i + 1, i + delay_bars + 1):
                            if pending_signal == 1 and data['close'].iloc[j] <= data[f'ema{self.ema_slow}'].iloc[j]:
                                confirmed = False
                                break
                            if pending_signal == -1 and data['close'].iloc[j] >= data[f'ema{self.ema_slow}'].iloc[j]:
                                confirmed = False
                                break
                        if confirmed:
                            entry_index = i + delay_bars
                            entry_time = data['time'].iloc[entry_index]
                            entry_price = data['close'].iloc[entry_index]
                            if pending_signal == 1:
                                stop_loss = entry_price * 0.99  # 1% below entry
                            else:
                                stop_loss = entry_price * 1.01  # 1% above entry
                            position_size = self.calculate_position_size(entry_price, stop_loss)
                            current_position = pending_signal
                            trade_type = "BUY" if pending_signal == 1 else "SELL"
                            self.trades.append({
                                'type': trade_type,
                                'entry_time': entry_time,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'position_size': position_size
                            })
                            print(f"{trade_type} signal confirmed at {entry_time}: Price={entry_price}, Size={position_size}, SL={stop_loss}")
                            i = entry_index
                            continue
            
            # For long positions, update trailing stop and check exit conditions
            elif current_position == 1:
                # Update trailing stop: move SL up if price moves higher
                new_stop = current_price * (1 - trailing_stop_pct)
                if new_stop > stop_loss:
                    stop_loss = new_stop
                    print(f"Updated trailing SL for long at {current_time}: new SL = {stop_loss}")
                if current_price <= stop_loss:
                    profit = (stop_loss - entry_price) * position_size * 100
                    self.capital += profit
                    self.trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': stop_loss,
                        'profit': profit,
                        'exit_reason': 'Stop Loss'
                    })
                    print(f"STOP LOSS at {current_time}: Price={stop_loss}, P&L=${profit:.2f}, Balance=${self.capital:.2f}")
                    current_position = 0
                elif current_signal == -1:
                    profit = (current_price - entry_price) * position_size * 100
                    self.capital += profit
                    self.trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'profit': profit,
                        'exit_reason': 'Crossover'
                    })
                    print(f"CLOSE BUY at {current_time}: Price={current_price}, P&L=${profit:.2f}, Balance=${self.capital:.2f}")
                    entry_price = current_price
                    stop_loss = entry_price * 1.01
                    position_size = self.calculate_position_size(entry_price, stop_loss)
                    current_position = -1
                    entry_time = current_time
                    self.trades.append({
                        'type': 'SELL',
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'position_size': position_size
                    })
                    print(f"SELL signal at {entry_time}: Price={entry_price}, Size={position_size}, SL={stop_loss}")
            
            # For short positions, update trailing stop and check exit conditions
            elif current_position == -1:
                new_stop = current_price * (1 + trailing_stop_pct)
                if new_stop < stop_loss:
                    stop_loss = new_stop
                    print(f"Updated trailing SL for short at {current_time}: new SL = {stop_loss}")
                if current_price >= stop_loss:
                    profit = (entry_price - stop_loss) * position_size * 100
                    self.capital += profit
                    self.trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': stop_loss,
                        'profit': profit,
                        'exit_reason': 'Stop Loss'
                    })
                    print(f"STOP LOSS at {current_time}: Price={stop_loss}, P&L=${profit:.2f}, Balance=${self.capital:.2f}")
                    current_position = 0
                elif current_signal == 1:
                    profit = (entry_price - current_price) * position_size * 100
                    self.capital += profit
                    self.trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'profit': profit,
                        'exit_reason': 'Crossover'
                    })
                    print(f"CLOSE SELL at {current_time}: Price={current_price}, P&L=${profit:.2f}, Balance=${self.capital:.2f}")
                    entry_price = current_price
                    stop_loss = entry_price * 0.99
                    position_size = self.calculate_position_size(entry_price, stop_loss)
                    current_position = 1
                    entry_time = current_time
                    self.trades.append({
                        'type': 'BUY',
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'position_size': position_size
                    })
                    print(f"BUY signal at {entry_time}: Price={entry_price}, Size={position_size}, SL={stop_loss}")
            
            i += 1

        # Close any open position at the end of the test
        if current_position != 0:
            final_price = data['close'].iloc[-1]
            if current_position == 1:
                profit = (final_price - entry_price) * position_size * 100
            else:
                profit = (entry_price - final_price) * position_size * 100
            self.capital += profit
            self.trades[-1].update({
                'exit_time': data['time'].iloc[-1],
                'exit_price': final_price,
                'profit': profit,
                'exit_reason': 'End of Test'
            })
            print(f"CLOSE POSITION at end: Price={final_price}, P&L=${profit:.2f}, Final Balance=${self.capital:.2f}")
        
        trades_df = pd.DataFrame(self.trades)
        self.calculate_performance(trades_df, data)
        return trades_df, data
        
    def calculate_performance(self, trades_df, price_data):
        """Calculate performance metrics"""
        if len(trades_df) == 0:
            print("No trades executed during the backtest period")
            return
        completed_trades = trades_df.dropna(subset=['exit_time'])
        total_trades = len(completed_trades)
        winning_trades = len(completed_trades[completed_trades['profit'] > 0])
        losing_trades = len(completed_trades[completed_trades['profit'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = completed_trades['profit'].sum()
        average_profit = completed_trades['profit'].mean()
        average_win = completed_trades[completed_trades['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        average_loss = completed_trades[completed_trades['profit'] <= 0]['profit'].mean() if losing_trades > 0 else 0
        profit_factor = abs(completed_trades[completed_trades['profit'] > 0]['profit'].sum() / 
                              completed_trades[completed_trades['profit'] <= 0]['profit'].sum()) if losing_trades > 0 else float('inf')
        
        equity_curve = pd.Series(self.equity_curve)
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        roi = ((self.capital - self.initial_capital) / self.initial_capital * 100)
        
        print("\n===== BACKTEST RESULTS =====")
        print(f"Testing Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital}")
        print(f"Final Capital: ${self.capital:.2f}")
        print(f"Total Return: ${self.capital - self.initial_capital:.2f} ({roi:.2f}%)")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades} ({win_rate:.2f}%)")
        print(f"Losing Trades: {losing_trades} ({100-win_rate:.2f}%)")
        print(f"Average Profit per Trade: ${average_profit:.2f}")
        print(f"Average Win: ${average_win:.2f}")
        print(f"Average Loss: ${average_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        
        self.performance_metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'average_profit': average_profit,
            'average_win': average_win,
            'average_loss': average_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'roi': roi
        }
        
        if 'entry_time' in completed_trades.columns:
            completed_trades['month'] = completed_trades['entry_time'].dt.strftime('%Y-%m')
            monthly_performance = completed_trades.groupby('month')['profit'].sum()
            print("\n===== MONTHLY PERFORMANCE =====")
            for month, profit in monthly_performance.items():
                print(f"{month}: ${profit:.2f}")
        
    def plot_results(self, price_data, trades_df):
        """Plot backtest results"""
        if price_data is None or len(price_data) == 0:
            print("No data to plot")
            return
        plt.figure(figsize=(14, 10))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(price_data['time'], price_data['close'], label='XAU/USD', color='black', alpha=0.5)
        ax1.plot(price_data['time'], price_data[f'ema{self.ema_fast}'], label=f'EMA{self.ema_fast}', color='blue')
        ax1.plot(price_data['time'], price_data[f'ema{self.ema_slow}'], label=f'EMA{self.ema_slow}', color='red')
        for _, trade in trades_df.iterrows():
            if 'entry_time' in trade and 'type' in trade:
                if trade['type'] == 'BUY':
                    ax1.scatter(trade['entry_time'], trade['entry_price'], marker='^', color='green', s=100)
                else:
                    ax1.scatter(trade['entry_time'], trade['entry_price'], marker='v', color='red', s=100)
            if 'exit_time' in trade and 'exit_price' in trade and not pd.isna(trade['exit_time']):
                ax1.scatter(trade['exit_time'], trade['exit_price'], marker='o', color='black', s=50)
        ax1.set_title(f'XAU/USD EMA Crossover Strategy (EMA{self.ema_fast} vs EMA{self.ema_slow})')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        date_format = DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(price_data['time'][:len(self.equity_curve)], self.equity_curve, label='Equity', color='green')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Capital ($)')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        ax2.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def save_results(self, trades_df, filename="backtest_results.csv"):
        """Save backtest results to CSV"""
        if trades_df is not None and len(trades_df) > 0:
            trades_df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        
    def run(self):
        """Run the complete backtest process"""
        try:
            trades_df, price_data = self.backtest()
            if trades_df is not None and price_data is not None and len(trades_df) > 0:
                self.plot_results(price_data, trades_df)
                self.save_results(trades_df)
            else:
                print("No trades or price data available for analysis")
        except Exception as e:
            print(f"Error in run method: {e}")
        finally:
            if self.mt5_connected:
                mt5.shutdown()
                print("MetaTrader 5 connection closed")
        return self.performance_metrics if hasattr(self, 'performance_metrics') else None

if __name__ == "__main__":
    start_date = datetime.now() - timedelta(days=730)
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    print(f"Testing date range: {start_date} to {end_date}")
    backtest = EMA_Crossover_Backtest(
        symbol="XAUUSD",
        timeframe=mt5.TIMEFRAME_M5,  # Adjust timeframe as needed
        ema_fast=80,
        ema_slow=200,
        risk_percent=1.0,
        initial_capital=100000,
        start_date=start_date,
        end_date=end_date
    )
    try:
        results = backtest.run()
        if results:
            print("Backtest completed successfully with results")
        else:
            print("Backtest completed but no results were generated")
    except Exception as e:
        print(f"Error running backtest: {e}")
