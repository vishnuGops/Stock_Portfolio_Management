import yfinance as yf
import pandas as pd

# Define a list of stock symbols
stock_symbols = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN"]

# Create an empty DataFrame to store the stock data
stock_data = pd.DataFrame(columns=["Symbol", "Price", "Change", "Change %"])

# Fetch data for each stock symbol and populate the DataFrame
for symbol in stock_symbols:
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d")

    if not data.empty:
        price = data["Close"].iloc[0]
        change = price - data["Open"].iloc[0]
        change_percent = (change / data["Open"].iloc[0]) * 100
        new_data = pd.DataFrame({"Symbol": [symbol], "Price": [price], "Change": [
                                change], "Change %": [change_percent]})
        stock_data = pd.concat([stock_data, new_data],
                               ignore_index=True, sort=False)

# Display the stock data in a table
print(stock_data)
