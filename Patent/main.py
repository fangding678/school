# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 10:20:21 2017

@author: fangding
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense


path = './datafile/'
filename = 'AllAdjCloseData.csv'
tickers = ["MSFT", "AAPL", "GE", "IBM", "AA", "DAL", "UAL", "PEP", "KO"]
sequence_len = 20
epoch = 5000
bsize = 64
split_rate = 0.95
evaluate_method = ['Mean Absolute Error', 'Mean Squared Error', 'Explained variance', \
                   'Coefficient of determination', 'My Idea']


def load_data(ticker):
    df = pd.read_csv(path+filename)
    arr = df[ticker]
    arr = np.array(arr).astype('float')
    date_index = df['Date']
    return date_index, arr


def tranform_data(date_index, arr):
    date_index = date_index[sequence_len+1:]
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(arr)
    raw_date = []
    for i in range(len(arr) - sequence_len - 1):
        raw_date.append(arr[i: i + sequence_len + 1])
    raw_date = np.array(raw_date).astype('float')
    split_bound = int(raw_date.shape[0]*split_rate)
    train_data = raw_date[:split_bound]
    test_data = raw_date[split_bound:]
    dindex_train = date_index[:split_bound]
    dindex_test = date_index[split_bound:]
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1]
    test_x = test_data[:, :-1]
    test_y = test_data[:, -1]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], -1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], -1))
    train_y = np.reshape(train_y, (train_y.shape[0], 1))
    test_y = np.reshape(test_y, (test_y.shape[0], 1))
    return dindex_train, dindex_test, train_x, train_y, test_x, test_y, scaler


def build_model():
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
    print(model.layers)
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def train_model(train_x, train_y, test_x, test_y, scaler):
    model = build_model()
    model.fit(train_x, train_y, batch_size=bsize, nb_epoch=epoch, validation_split=0.2)
    y_predict = model.predict(test_x)
    #print(y_predict, test_y)
    y_predict = scaler.inverse_transform(y_predict)
    test_y = scaler.inverse_transform(test_y)
    #print(y_predict, test_y)
    return y_predict, test_y


def evaluate(y_predict, test_y, method):
    y_predict = np.ravel(y_predict)
    test_y = np.ravel(test_y)
    print(y_predict)
    if evaluate_method[0] in method:
        error = np.abs(y_predict - test_y).mean()
        print('the loss of %s is %f' % (evaluate_method[0], error))
    if evaluate_method[1] in method:
        error = np.dot(y_predict - test_y, y_predict - test_y) / len(y_predict)
        print('the loss of %s is %f' % (evaluate_method[1], error))
    if evaluate_method[2] in method:
        error = 1 - np.var(y_predict - test_y)/np.var(test_y)
        print('the loss of %s is %f' % (evaluate_method[2], error))
    if evaluate_method[3] in method:
        error = 1 - np.dot(y_predict - test_y, y_predict - test_y) / \
                    np.dot(test_y - np.mean(test_y), test_y - np.mean(test_y))
        print('the loss of %s is %f' % (evaluate_method[3], error))
    if evaluate_method[4] in method:
        error = np.mean(np.abs(y_predict - test_y) / test_y)
        print('the loss of %s is %f' % (evaluate_method[4], error))


def result_plot(y_predict, test_y, dindex_test, ind):
    #print(len(dindex_test))
    #print(len(y_predict))
    #print(len(test_y))
    '''
    plt.figure(1)
    plt.plot(dindex_test, y_predict, 'r--')
    plt.plot(dindex_test, test_y, 'g-')
    plt.legend(['predict', 'true'])
    plt.show()
    '''
    y_predict = np.ravel(y_predict)
    test_y = np.ravel(test_y)
    date_label = pd.to_datetime(dindex_test)
    s_predict = pd.Series(y_predict, index=date_label)
    s_true = pd.Series(test_y, index=date_label)
    df = pd.DataFrame(index=s_true.index)
    df['predict'] = s_predict
    df['true'] = s_true
    plt.figure(figsize=(12, 8))
    df.plot()
    plt.legend(loc='best')
    plt.ylim(45, 75)
    pfile = './resultfile/picture_result' + str(epoch) + '_' + str(ind) + '.png'
    plt.savefig(pfile)
    plt.show()
    dfw = pd.DataFrame([y_predict, test_y])
    dfw = dfw.T
    dfw.columns=['predict', 'true']
    tfile = './resultfile/text_result' + str(epoch) + '_' + str(ind) + '.csv'
    df.to_csv(tfile)



if __name__ == '__main__':
    s = time.time()
    date_index, arr = load_data(tickers[0])
    for i in range(10):
        dindex_train, dindex_test, train_x, train_y, test_x, test_y, scaler = tranform_data(date_index, arr)
        y_predict, test_y = train_model(train_x, train_y, test_x, test_y, scaler)
        evaluate(y_predict, test_y, evaluate_method)
        result_plot(y_predict, test_y, dindex_test, i)
    e = time.time()
    print('the total of time: ', str(e-s), 's')








