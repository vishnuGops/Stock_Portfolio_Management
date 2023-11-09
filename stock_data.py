# IMPORT THE LIBRARY
import yfinance as yf
from datetime import datetime
import finplot as fplt


def get_data():
    # CREATE TICKER INSTANCE FOR AMAZON
    amzn = yf.Ticker("TSLA")

    # GET TODAYS DATE AND CONVERT IT TO A STRING WITH YYYY-MM-DD FORMAT (YFINANCE EXPECTS THAT FORMAT)
    end_date = datetime.now().strftime('%Y-%m-%d')
    amzn_hist = amzn.history(start='2018-01-01', end=end_date)
    print(amzn_hist)


def plot_graph(ticker):
    # RETRIEVE 1 YEAR WORTH OF DAILY DATA OF TESLA
    end_date = datetime.now().strftime('%Y-%m-%d')
    df = ticker.history(interval='1d', period='1y')

    # PLOT THE OHLC CANDLE CHART
    fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
    fplt.show()


def main():
    get_data()
    plot_graph(yf.Ticker('SPY'))


if __name__ == "__main__":
    main()
