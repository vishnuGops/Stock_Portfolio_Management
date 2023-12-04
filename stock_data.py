import yfinance as yf


def get_stock_data(ticker):
    # Fetch stock data from Yahoo Finance
    stock_data = yf.download(ticker, period="5d", interval="1d")

    # Calculate some basic statistics
    average_price = stock_data['Close'].mean()
    standard_deviation = stock_data['Close'].std()
    moving_average = stock_data['Close'].rolling(window=20).mean()
    macd = stock_data['Close'].ewm(span=12, min_periods=9).mean(
    ) - stock_data['Close'].ewm(span=26, min_periods=25).mean()
    rsi = stock_data['Close'].diff(1).ewm(span=14).mean(
    ) / stock_data['Close'].diff(1).ewm(span=14, min_periods=13).mean() * 100

    # Generate a summary of the stock data
    summary = {
        "Ticker": ticker,
        "Average Price": average_price,
        "Standard Deviation": standard_deviation,
        "Moving Average (20-day)": moving_average.iloc[-1],
        "MACD": macd.iloc[-1],
        "RSI": rsi.iloc[-1]
    }

    return summary


def main():
    ticker = input("Enter a stock ticker: ")
    stock_data = get_stock_data(ticker)

    print("Stock Summary:")
    for key, value in stock_data.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
