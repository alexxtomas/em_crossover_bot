# EMA Crossover Trading Bot
source venv/Scripts/activate

This is an automated trading bot that uses Exponential Moving Average (EMA) crossovers to generate trading signals for the XAUUSD (Gold) market using MetaTrader 5.

## Requirements

- Python 3.8+
- MetaTrader 5 installed on your computer
- Active trading account connected to MetaTrader 5

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure MetaTrader 5 is installed and running on your computer

### Important Note for macOS Users

MetaTrader 5 is primarily designed for Windows and does not have a native macOS version. To use this bot on macOS, you have several options:

1. **Use a Windows Virtual Machine**: Install Windows on a virtual machine using software like Parallels, VMware, or VirtualBox, then install MetaTrader 5 on the virtual machine.

2. **Use Wine**: Wine is a compatibility layer that allows Windows applications to run on macOS. You can try installing MetaTrader 5 using Wine, but functionality may be limited.

3. **Use a VPS (Virtual Private Server)**: Rent a Windows VPS and run MetaTrader 5 and this bot on the VPS. This has the advantage of running 24/7 without needing to keep your computer on.

After setting up MetaTrader 5 on Windows (via any of the methods above), you'll need to install the MetaTrader5 Python package:

```bash
pip install MetaTrader5==5.0.45
```

## Configuration

The bot is configured with the following default parameters:

- Symbol: XAUUSD (Gold)
- Timeframe: 5-minute candles
- Fast EMA: 80 periods
- Slow EMA: 280 periods
- Risk per trade: 1% of account balance

You can modify these parameters in the `main.py` file.

## Usage

Run the bot with:

```bash
python main.py
```

The bot will:

- Connect to MetaTrader 5
- Calculate EMA crossovers
- Generate buy signals when the fast EMA crosses above the slow EMA
- Generate sell signals when the fast EMA crosses below the slow EMA
- Automatically manage position sizing based on your risk settings
- Place and manage trades with appropriate stop losses

Press Ctrl+C to stop the bot.

## Important Notes

- This bot requires a running instance of MetaTrader 5 with an active account
- Always test on a demo account before using real funds
- Past performance is not indicative of future results
