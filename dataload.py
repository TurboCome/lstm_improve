
import matplotlib.pyplot as plt
from keras.models import Sequential
# from keras.optimizers import adam_v2
# adam = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
from keras.optimizers import Adam
from metrics import *
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout,LSTM,RNN,SimpleRNN
import numpy as np
import pandas as pd


def load_data(data_path, columns):
    df = pd.read_csv(data_path, encoding='gbk')
    new_df = df[columns]  # 'Volume'

    new_df.fillna(new_df.mean(), inplace=True)
    return new_df


def split_data(df, time_stamp, n_skip):

    # 划分训练集与验证集
    index = round(len(df) * 0.8)
    google_stock = df[['e', 'c', 'h', 'g', 'h2']]  # 'Volume'
    train = google_stock[:index + time_stamp]
    test = google_stock[index - time_stamp:]
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train)
    x_train, y_train = [], []
    # 训练集
    print(scaled_data.shape)
    print(scaled_data[1, 3])
    for i in range(time_stamp, len(train)):
        x_train.append(scaled_data[i - time_stamp:i])
        y_train.append(scaled_data[i, 0])  # y = 'e'
    x_train, y_train = np.array(x_train), np.array(y_train)

    # 验证集
    scaled_data = scaler.fit_transform(test)
    x_test, y_test = [], []
    for i in range(time_stamp, len(test)):
        x_test.append(scaled_data[i - time_stamp:i])
        y_test.append(scaled_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    print(x_train.shape)
    print(x_test.shape)
    return x_train,y_train,x_test,y_test, test


# 划分对未来预测的数据：  30组，预测 1组
def createXY(df, n_past, n_skip):
    # 划分训练集与验证集
    index = round(len(df) * 0.8)
    train = df[:index + n_past]
    test = df[index - n_past:]
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    x_train, y_train = [], []

    for i in range(n_past, len(train_scaled)):
          x_train.append(train_scaled[i - n_past:i, 0:train_scaled.shape[1]])
          y_train.append(train_scaled[i:i+n_skip,0])
    x_train,y_train = np.array(x_train),np.array(y_train)


    test_scaled = scaler.fit_transform(test)
    x_test, y_test = [], []
    for i in range(n_past, len(test_scaled)):
        x_test.append(test_scaled[i-n_past:i, 0:test_scaled.shape[1]])
        y_test.append(test_scaled[i:i+n_skip,0])
    x_test,y_test = np.array(x_test), np.array(y_test)

    return x_train,y_train,x_test,y_test


def custom_ts_multi_data_prep(df, window, horizon, idx):
    index = round(len(df) * 0.8)
    train = df[ :index + window]
    test = df[index - window: ]
    # 归一化 train: (None, 5)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(train)
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.fit_transform(test)

    y_data = train_scaled[:,idx]
    x_train,y_train = [], []
    start,end = (0 + window), (index+window-horizon)
    for i in range(start, end):
        indicex = range(i - window, i)
        x_train.append(train_scaled[indicex])
        indicey = range(i , i + horizon)
        y_train.append(y_data[indicey])

    y_data = test_scaled[:,idx]
    x_test, y_test = [], []
    start, end = (0 + window), (len(df)-index-horizon)
    for i in range(start, end):
        indicex = range(i - window, i)
        x_test.append(test_scaled[indicex])
        indicey = range(i, i + horizon)
        y_test.append(y_data[indicey])

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), scaler



def create_batch_data(x_train,y_train,x_test,y_test, batch_size):
    x_train_batch,y_train_batch = [],[]
    x_test_batch, y_test_batch = [], []

    for i in range(batch_size,len(x_train)-batch_size):
        x_train_batch.append(x_train[i:i+batch_size,:])
        y_train_batch.append(y_train[i:i+batch_size,:])

    for i in range(batch_size,len(x_test)-batch_size):
        x_test_batch.append(x_test[i:i+batch_size,:])
        y_test_batch.append(y_test[i:i+batch_size,:])

    return np.array(x_train_batch), np.array(y_train_batch),np.array(x_test_batch), np.array(y_test_batch)