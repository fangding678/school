import pandas as pd
from pandas_datareader import data
from datetime import datetime


start = datetime(2015, 1, 1)
end = datetime(2016, 12, 31)
path = './datafile/'
tickers = ["MSFT", "AAPL", "GE", "IBM", "AA", "DAL", "UAL", "PEP", "KO"]


def get_stock_data(ticker):
    da = data.DataReader(ticker, 'yahoo', start, end)
    da.rename(columns={'Adj Close': 'AdjClose'}, inplace=True)
    da.insert(0, 'Ticker', ticker)
    return da


def get_all_stock_data():
    rawStocks = {}
    for ticker in tickers:
        rawStocks[ticker] = get_stock_data(ticker)
    return rawStocks


def pivot_attribute_to_columns(column='AdjClose'):
    items = []
    rawStocks = get_all_stock_data()
    for key, value in rawStocks.items():
        subset = value[['Ticker', column]]
        items.append(subset)
    combined = pd.concat(items)
    print(combined)
    ri = combined.reset_index()
    print(ri)
    df = ri.pivot("Date", "Ticker", column)
    df.to_csv(path + 'All' + column + 'Data.csv')
    return df


def write_data_to_file():
    for ticker in tickers:
        s = get_stock_data(ticker)
        pfile = path + ticker + '.csv'
        s.to_csv(pfile)


def print_instance():
    for ticker in tickers:
        print(ticker + ': ' + str(len(get_stock_data(ticker))))


if __name__ == '__main__':
    #write_data_to_file()
    #print_instance()
    #df = pivot_attribute_to_columns()
    pass





