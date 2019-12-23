# -*- coding: uft-8 -*-


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


path = './datafile/'
filename = 'AllAdjCloseData.csv'
tickers = ["MSFT", "AAPL", "GE", "IBM", "AA", "DAL", "UAL", "PEP", "KO"]


def load_data(ticker):
    df = pd.read_csv(path+filename, index_col='Date')
    df = df[[ticker]]
    print(df)
    return df


def paint(df):
    df.plot()
    plt.show()


if __name__ == '__main__':
    df = load_data(tickers[0])
    paint(df)