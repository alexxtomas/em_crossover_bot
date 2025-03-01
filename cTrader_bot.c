using System;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;
using System.Collections.Generic;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class EMAcrossoverStrategy : Robot
    {
        [Parameter("Fast EMA Period", DefaultValue = 80)]
        public int FastEmaPeriod { get; set; }

        [Parameter("Slow EMA Period", DefaultValue = 200)]
        public int SlowEmaPeriod { get; set; }

        [Parameter("Risk Percent", DefaultValue = 1.0)]
        public double RiskPercent { get; set; }

        [Parameter("Confirmation Delay (Bars)", DefaultValue = 3)]
        public int ConfirmationDelay { get; set; }

        [Parameter("Trailing Stop (%)", DefaultValue = 0.5)]
        public double TrailingStopPercent { get; set; }

        [Parameter("Initial Stop Loss (%)", DefaultValue = 1.0)]
        public double InitialStopLossPercent { get; set; }

        private ExponentialMovingAverage _fastEma;
        private ExponentialMovingAverage _slowEma;
        private int _lastBarProcessed = -1;
        private Queue<PendingSignal> _pendingSignals = new Queue<PendingSignal>();
        private Dictionary<string, double> _trailingStops = new Dictionary<string, double>();

        protected override void OnStart()
        {
            _fastEma = Indicators.ExponentialMovingAverage(Bars.ClosePrices, FastEmaPeriod);
            _slowEma = Indicators.ExponentialMovingAverage(Bars.ClosePrices, SlowEmaPeriod);
            
            // Initialize with historical data
            int startBar = Math.Max(FastEmaPeriod, SlowEmaPeriod) + 10;
            for (int i = startBar; i < Bars.Count - 1; i++)
            {
                ProcessBar(i);
            }
        }

        protected override void OnBar()
        {
            int currentBar = Bars.Count - 1;
            if (_lastBarProcessed == currentBar)
                return;

            ProcessBar(currentBar);
            _lastBarProcessed = currentBar;
        }

        protected override void OnTick()
        {
            // Update trailing stops for open positions
            foreach (var position in Positions)
            {
                if (!_trailingStops.ContainsKey(position.Label))
                    continue;

                double currentTrailingStop = _trailingStops[position.Label];
                
                if (position.TradeType == TradeType.Buy)
                {
                    double newStop = Symbol.Bid * (1 - TrailingStopPercent / 100);
                    if (newStop > currentTrailingStop)
                    {
                        _trailingStops[position.Label] = newStop;
                        if (position.StopLoss < newStop)
                        {
                            ModifyPosition(position, newStop, position.TakeProfit);
                            Print($"Updated trailing SL for long position {position.Label}: new SL = {newStop}");
                        }
                    }
                }
                else if (position.TradeType == TradeType.Sell)
                {
                    double newStop = Symbol.Ask * (1 + TrailingStopPercent / 100);
                    if (newStop < currentTrailingStop)
                    {
                        _trailingStops[position.Label] = newStop;
                        if (position.StopLoss > newStop)
                        {
                            ModifyPosition(position, newStop, position.TakeProfit);
                            Print($"Updated trailing SL for short position {position.Label}: new SL = {newStop}");
                        }
                    }
                }
            }
        }

        private void ProcessBar(int barIndex)
        {
            // Check for signal
            if (barIndex <= SlowEmaPeriod)
                return;

            bool bullishCrossover = _fastEma.Result[barIndex] > _slowEma.Result[barIndex] && 
                                  _fastEma.Result[barIndex - 1] <= _slowEma.Result[barIndex - 1];
            
            bool bearishCrossover = _fastEma.Result[barIndex] < _slowEma.Result[barIndex] && 
                                   _fastEma.Result[barIndex - 1] >= _slowEma.Result[barIndex - 1];

            if (bullishCrossover || bearishCrossover)
            {
                TradeType signalType = bullishCrossover ? TradeType.Buy : TradeType.Sell;
                PendingSignal signal = new PendingSignal
                {
                    Type = signalType,
                    BarIndex = barIndex,
                    ConfirmationBarIndex = barIndex + ConfirmationDelay
                };
                
                _pendingSignals.Enqueue(signal);
                Print($"{signalType} signal detected at bar {barIndex}, confirmation at {signal.ConfirmationBarIndex}");
            }

            // Process pending signals
            ProcessPendingSignals(barIndex);
        }

        private void ProcessPendingSignals(int currentBarIndex)
        {
            if (_pendingSignals.Count == 0)
                return;

            // Check the oldest signal
            PendingSignal signal = _pendingSignals.Peek();
            
            // If not yet time for confirmation
            if (currentBarIndex < signal.ConfirmationBarIndex)
                return;
                
            // Time to check confirmation
            _pendingSignals.Dequeue();
            
            // Confirm signal
            bool confirmed = true;
            TradeType signalType = signal.Type;
            
            // Check confirmation period
            for (int i = signal.BarIndex + 1; i <= signal.ConfirmationBarIndex; i++)
            {
                if (signalType == TradeType.Buy && Bars.ClosePrices[i] <= _slowEma.Result[i])
                {
                    confirmed = false;
                    break;
                }
                else if (signalType == TradeType.Sell && Bars.ClosePrices[i] >= _slowEma.Result[i])
                {
                    confirmed = false;
                    break;
                }
            }

            if (!confirmed)
            {
                Print($"{signalType} signal not confirmed");
                return;
            }

            // Signal confirmed - execute trade
            ExecuteTrade(signalType);
        }

        private void ExecuteTrade(TradeType tradeType)
        {
            // Close any existing positions in opposite direction
            foreach (var position in Positions)
            {
                if (position.TradeType != tradeType)
                {
                    ClosePosition(position);
                    Print($"Closed {position.TradeType} position due to new {tradeType} signal");
                }
            }

            // Calculate entry price
            double entryPrice = tradeType == TradeType.Buy ? Symbol.Ask : Symbol.Bid;

            // Calculate stop loss
            double stopLoss;
            if (tradeType == TradeType.Buy)
                stopLoss = entryPrice * (1 - InitialStopLossPercent / 100);
            else
                stopLoss = entryPrice * (1 + InitialStopLossPercent / 100);

            // Calculate position size
            double positionSize = CalculatePositionSize(entryPrice, stopLoss);

            // Execute trade
            string label = $"EMA_{FastEmaPeriod}_{SlowEmaPeriod}_{Server.Time.Ticks}";
            ExecuteMarketOrder(tradeType, Symbol.Name, positionSize, label, stopLoss, null);
            _trailingStops[label] = stopLoss;

            Print($"Executed {tradeType} order: Price={entryPrice}, Size={positionSize}, SL={stopLoss}");
        }

        private double CalculatePositionSize(double entryPrice, double stopLoss)
        {
            double riskAmount = Account.Balance * (RiskPercent / 100);
            double priceDifference = Math.Abs(entryPrice - stopLoss);
            
            // Check for zero price difference (unlikely but safe to check)
            if (priceDifference == 0)
                return Symbol.VolumeInUnitsMin;
                
            double riskPerLot = priceDifference * Symbol.TickValue / Symbol.TickSize;
            double positionSize = riskAmount / riskPerLot;
            
            // Round to standard lot sizes and ensure minimum
            positionSize = Math.Floor(positionSize * 100) / 100; // Round to 2 decimal places
            positionSize = Math.Max(positionSize, Symbol.VolumeInUnitsMin);
            positionSize = Math.Min(positionSize, Symbol.VolumeInUnitsMax);
            
            return positionSize;
        }

        private class PendingSignal
        {
            public TradeType Type { get; set; }
            public int BarIndex { get; set; }
            public int ConfirmationBarIndex { get; set; }
        }
    }
}