import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class StockBacktester:
    def __init__(self, symbol, start_date, end_date, initial_capital=10000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = None
        self.trades = []
        self.portfolio_value = []
        self.current_position = 0  # 0: no position, 1: long position
        self.cash = initial_capital
        self.equity = 0
        self.shares_held = 0
        self.entry_price = 0

    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(
            f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}")
        self.data = yf.download(
            self.symbol, start=self.start_date, end=self.end_date)
        if self.data.empty:
            raise ValueError(
                "No data fetched. Check your symbol and date range.")
        print(f"Fetched {len(self.data)} data points")
        return self.data

    def calculate_indicators(self):
        """Calculate technical indicators"""
        # Calculate VWAP (Volume Weighted Average Price)
        self.data['VWAP'] = (self.data['Close'] * self.data['Volume']
                             ).cumsum() / self.data['Volume'].cumsum()

        # Calculate EMAs
        self.data['EMA9'] = self.data['Close'].ewm(span=9, adjust=False).mean()
        self.data['EMA21'] = self.data['Close'].ewm(
            span=21, adjust=False).mean()

        # Calculate RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # Calculate additional metrics for strategy evaluation
        self.data['Returns'] = self.data['Close'].pct_change()

        # Check that all indicators exist
        expected_columns = ['VWAP', 'EMA9', 'EMA21', 'RSI', 'Returns']
        for col in expected_columns:
            if col not in self.data.columns:
                print(f"Warning: {col} column was not created")

        return self.data

    def generate_signals(self):
        """Generate buy/sell signals based on EMA crossover strategy"""
        # Initialize signal columns
        self.data['Signal'] = 0  # 0: no signal, 1: buy signal, -1: sell signal

        # EMA crossover strategy
        self.data['EMA_Diff'] = self.data['EMA9'] - self.data['EMA21']
        self.data['EMA_Diff_Prev'] = self.data['EMA_Diff'].shift(1)

        # Buy signal: EMA9 crosses above EMA21
        buy_signal = (self.data['EMA_Diff'] > 0) & (
            self.data['EMA_Diff_Prev'] <= 0)
        self.data.loc[buy_signal, 'Signal'] = 1

        # Sell signal: EMA9 crosses below EMA21
        sell_signal = (self.data['EMA_Diff'] < 0) & (
            self.data['EMA_Diff_Prev'] >= 0)
        self.data.loc[sell_signal, 'Signal'] = -1

        # Add RSI filter (optional)
        # Only buy if RSI < 70 (not overbought)
        self.data.loc[(self.data['Signal'] == 1) & (
            self.data['RSI'] >= 70), 'Signal'] = 0

        # Only sell if RSI > 30 (not oversold)
        self.data.loc[(self.data['Signal'] == -1) &
                      (self.data['RSI'] <= 30), 'Signal'] = 0

        return self.data

    def backtest_strategy(self):
        """Run backtest on the generated signals with take profit at 3% gain"""
        # Initialize portfolio tracking
        self.portfolio_value = []
        self.cash = self.initial_capital
        self.equity = 0
        self.current_position = 0
        self.shares_held = 0
        self.entry_price = 0
        self.trades = []

        # Skip the initial period where indicators are not fully calculated (typically need at least 21 days for EMA21)
        start_index = 22

        # Initialize the portfolio values for the skipped days
        for i in range(start_index):
            self.portfolio_value.append(self.initial_capital)

        # Add a column to track if take profit was triggered
        self.data['Take_Profit'] = 0

        for i in range(start_index, len(self.data)):
            current_idx = self.data.index[i]
            prev_idx = self.data.index[i-1] if i > start_index else None
            row = self.data.iloc[i]
            current_close = float(row['Close'].iloc[0])

            # Check for take profit if we have a position
            take_profit = False
            if self.current_position > 0 and self.entry_price > 0:
                current_gain = (
                    current_close - self.entry_price) / self.entry_price

                # Take profit at 3% gain by selling 50% of position
                if current_gain >= 0.03:
                    take_profit = True
                    self.data.loc[current_idx, 'Take_Profit'] = 1

                    # Calculate shares to sell (50% of position)
                    shares_to_sell = self.shares_held // 2
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * current_close
                        self.cash += proceeds
                        self.shares_held -= shares_to_sell
                        self.equity = self.shares_held * current_close

                        # Record the trade
                        self.trades.append({
                            'Date': current_idx,
                            'Type': 'TAKE_PROFIT',
                            'Price': current_close,
                            'Shares': shares_to_sell,
                            'Value': proceeds,
                            'Portfolio': self.cash + self.equity
                        })

            # Update equity value based on current price
            if self.shares_held > 0:
                self.equity = self.shares_held * current_close

            # Process signals
            signal = float(row['Signal'].iloc[0])

            # Process buy signal
            if signal == 1 and self.cash > 0:
                # Calculate maximum shares we can buy with available cash
                max_shares = self.cash // current_close
                if max_shares > 0:
                    cost = max_shares * current_close
                    self.cash -= cost
                    self.shares_held += max_shares
                    self.equity = self.shares_held * current_close
                    self.current_position = 1
                    self.entry_price = current_close

                    self.trades.append({
                        'Date': current_idx,
                        'Type': 'BUY',
                        'Price': current_close,
                        'Shares': max_shares,
                        'Value': cost,
                        'Portfolio': self.cash + self.equity
                    })

            # Process sell signal
            elif signal == -1 and self.shares_held > 0:
                proceeds = self.shares_held * current_close
                self.cash += proceeds

                self.trades.append({
                    'Date': current_idx,
                    'Type': 'SELL',
                    'Price': current_close,
                    'Shares': self.shares_held,
                    'Value': proceeds,
                    'Portfolio': self.cash + self.equity
                })

                self.shares_held = 0
                self.equity = 0
                self.current_position = 0
                self.entry_price = 0

            # Append current portfolio value
            self.portfolio_value.append(self.cash + self.equity)

        # Add portfolio value to dataframe
        portfolio_series = pd.Series(
            self.portfolio_value, index=self.data.index)
        self.data['Portfolio'] = portfolio_series

        return self.trades

    def calculate_performance_metrics(self):
        """Calculate performance metrics for the strategy"""
        if len(self.portfolio_value) < 2:
            return {"error": "Not enough data to calculate performance metrics"}

        # Total return
        total_return = (
            self.portfolio_value[-1] - self.initial_capital) / self.initial_capital * 100

        # Annualized return
        days = (self.data.index[-1] - self.data.index[0]).days
        annualized_return = ((1 + total_return / 100) **
                             (365 / days) - 1) * 100 if days > 0 else 0

        # Make sure portfolio values are numeric
        self.data['Portfolio'] = pd.to_numeric(
            self.data['Portfolio'], errors='coerce')

        # Daily returns - ensure it's numeric
        self.data['Strategy_Returns'] = pd.to_numeric(
            self.data['Portfolio'].pct_change(), errors='coerce')

        # Drop NaN values before calculations
        strategy_returns = self.data['Strategy_Returns'].dropna()

        # Calculate Sharpe Ratio (assuming risk-free rate of 0)
        mean_return = float(strategy_returns.mean()
                            ) if not strategy_returns.empty else 0
        std_return = float(strategy_returns.std()
                           ) if not strategy_returns.empty else 1

        if std_return == 0 or pd.isna(std_return):
            sharpe_ratio = 0
        else:
            sharpe_ratio = np.sqrt(252) * mean_return / std_return

        # Maximum Drawdown - ensure we're using numeric values
        portfolio_values = pd.to_numeric(
            self.data['Portfolio'], errors='coerce')
        cumulative_max = portfolio_values.cummax()
        self.data['Cumulative_Max'] = cumulative_max

        # Avoid division by zero
        valid_indices = (self.data['Cumulative_Max'] > 0) & (
            ~self.data['Cumulative_Max'].isna())
        self.data['Drawdown'] = 0.0
        if valid_indices.any():
            self.data.loc[valid_indices, 'Drawdown'] = (
                (self.data.loc[valid_indices, 'Portfolio'] - self.data.loc[valid_indices, 'Cumulative_Max']) /
                self.data.loc[valid_indices, 'Cumulative_Max']) * 100

        max_drawdown = float(self.data['Drawdown'].min(
        )) if not self.data['Drawdown'].empty else 0

        # Count trade types
        buy_trades = len([t for t in self.trades if t['Type'] == 'BUY'])
        sell_trades = len([t for t in self.trades if t['Type'] == 'SELL'])
        take_profit_trades = len(
            [t for t in self.trades if t['Type'] == 'TAKE_PROFIT'])

        # Calculate win rate and other trade metrics
        profitable_trades = 0
        total_profit = 0
        total_loss = 0
        profits = []
        losses = []

        # Analyze trades
        i = 0
        while i < len(self.trades):
            if self.trades[i]['Type'] == 'BUY':
                entry_price = self.trades[i]['Price']
                entry_shares = self.trades[i]['Shares']
                total_cost = entry_shares * entry_price
                total_proceeds = 0
                remaining_shares = entry_shares

                # Find all corresponding sells and take profits
                j = i + 1
                while j < len(self.trades) and remaining_shares > 0 and self.trades[j]['Type'] in ['SELL', 'TAKE_PROFIT']:
                    exit_price = self.trades[j]['Price']
                    exit_shares = min(remaining_shares,
                                      self.trades[j]['Shares'])
                    proceeds = exit_shares * exit_price

                    # Calculate trade profit for this portion
                    portion_cost = (exit_shares / entry_shares) * total_cost
                    trade_profit = proceeds - portion_cost
                    profit_pct = (trade_profit / portion_cost) * 100

                    if trade_profit > 0:
                        profitable_trades += 1
                        total_profit += trade_profit
                        profits.append(profit_pct)
                    else:
                        total_loss += trade_profit
                        losses.append(profit_pct)

                    remaining_shares -= exit_shares
                    total_proceeds += proceeds
                    j += 1

                # Skip to after the last sell/take profit
                i = j
            else:
                i += 1

        total_trades = len(profits) + len(losses)
        win_rate = (profitable_trades / total_trades) * \
            100 if total_trades > 0 else 0
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_trade = (sum(profits) + sum(losses)) / \
            total_trades if total_trades > 0 else 0

        return {
            "Total Return (%)": total_return,
            "Annualized Return (%)": annualized_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown (%)": max_drawdown,
            "Win Rate (%)": win_rate,
            "Average Trade Profit (%)": avg_trade,
            "Average Win (%)": avg_profit,
            "Average Loss (%)": avg_loss,
            "Number of Buy Trades": buy_trades,
            "Number of Sell Trades": sell_trades,
            "Number of Take Profit Trades": take_profit_trades,
            "Total Trades": total_trades,
            "Buy and Hold Return (%)": (self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0] * 100
        }

    def plot_results(self):
        """Plot trading results with buy/sell markers on candlestick chart"""
        # Create a subplot with 3 rows
        fig = make_subplots(rows=3, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=('Price & Trades',
                                            'Portfolio Value', 'RSI'),
                            row_heights=[0.6, 0.2, 0.2])

        # Add candlestick chart - ensure using OHLC data correctly
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Add EMAs
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['EMA9'],
                name='EMA9',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['EMA21'],
                name='EMA21',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )

        # Add VWAP
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['VWAP'],
                name='VWAP',
                line=dict(color='purple', width=1, dash='dash')
            ),
            row=1, col=1
        )

        # Find buy signals directly from trades
        buy_dates = [trade['Date']
                     for trade in self.trades if trade['Type'] == 'BUY']
        buy_prices = [self.data.loc[date, 'Low'] * 0.99 for date in buy_dates]

        if buy_dates:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='green',
                        line=dict(width=1, color='green')
                    ),
                    name='Buy Signal'
                ),
                row=1, col=1
            )

        # Find sell signals directly from trades
        sell_dates = [trade['Date']
                      for trade in self.trades if trade['Type'] == 'SELL']
        sell_prices = [self.data.loc[date, 'High']
                       * 1.01 for date in sell_dates]

        if sell_dates:
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='red',
                        line=dict(width=1, color='red')
                    ),
                    name='Sell Signal'
                ),
                row=1, col=1
            )

        # Find take profit signals directly from trades
        tp_dates = [trade['Date']
                    for trade in self.trades if trade['Type'] == 'TAKE_PROFIT']
        tp_prices = [self.data.loc[date, 'High'] * 1.01 for date in tp_dates]

        if tp_dates:
            fig.add_trace(
                go.Scatter(
                    x=tp_dates,
                    y=tp_prices,
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=12,
                        color='orange',
                        line=dict(width=1, color='orange')
                    ),
                    name='Take Profit'
                ),
                row=1, col=1
            )

        # Add portfolio value
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Portfolio'],
                name='Portfolio Value',
                line=dict(color='green', width=1.5)
            ),
            row=2, col=1
        )

        # Add buy and hold benchmark
        benchmark = self.data['Close'] / \
            self.data['Close'].iloc[0] * self.initial_capital
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=benchmark,
                name='Buy & Hold',
                line=dict(color='gray', width=1, dash='dot')
            ),
            row=2, col=1
        )

        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['RSI'],
                name='RSI',
                line=dict(color='blue', width=1)
            ),
            row=3, col=1
        )

        # Add RSI overbought/oversold lines
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=[70] * len(self.data.index),
                name='Overbought',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=[30] * len(self.data.index),
                name='Oversold',
                line=dict(color='green', width=1, dash='dash')
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title='Trading Strategy Backtest Results with Take Profit at 3%',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=800,
            width=1200
        )

        # Update y-axis for RSI
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

        # Update y-axis for Portfolio
        fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)

        # Update y-axis for Price
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)

        # Ensure candlesticks render properly by updating the range slider
        fig.update_xaxes(
            rangeslider_visible=False,
            row=1, col=1
        )

        return fig


# Main execution function
def run_backtest(symbol='SPY', start_date='2020-01-01', end_date='2023-12-31', initial_capital=10000):
    """Run the complete backtest process and return results"""
    # Initialize backtester
    backtester = StockBacktester(symbol, start_date, end_date, initial_capital)

    # Fetch data
    backtester.fetch_data()

    # Calculate indicators
    backtester.calculate_indicators()

    # Generate signals
    backtester.generate_signals()

    # Run backtest
    backtester.backtest_strategy()

    # Calculate performance metrics
    metrics = backtester.calculate_performance_metrics()

    # Plot results
    fig = backtester.plot_results()

    return backtester, metrics, fig


# Example usage
if __name__ == "__main__":
    # Set parameters
    symbol = 'SPY'
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    initial_capital = 10000

    # Run backtest
    backtester, metrics, fig = run_backtest(
        symbol, start_date, end_date, initial_capital)

    # Print performance metrics
    print("\n=== Performance Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(
            value, float) else f"{key}: {value}")

    # Show the plot
    fig.show()

    # You can also save the plot as HTML
    # fig.write_html("spy_backtest_results.html")
