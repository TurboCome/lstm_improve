from datetime import timedelta
from metrics import *
from pandas import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import statsmodels.api as sm


def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]

def data_load():
    data_path = r'/Users/whq/Public/codeDemo/lstm_improve/data/dealed_data.csv'
    df = pd.read_csv(data_path, encoding='gbk', index_col='date')
    #时间序列索引转换为日期格式
    df.index = pd.to_datetime(df.index)
    #指标量转为float类型
    df[['e', 'c', 'h', 'g', 'h2']] = df[['e', 'c', 'h', 'g', 'h2']].astype(float)

    # plt.figure(facecolor='white',figsize=(20,8))
    # plt.plot(df.index, df['e'],label='Time Series')
    # plt.legend(loc='best')
    # plt.show()
    return df

# 变量间相关性分析
def calculate_val(df):
    index = round(len(df)*0.8)
    pred_day = 7
    train_start = datetime(2002,1,1)
    train_end = datetime(2018,1,1)

    pred_start = train_end+timedelta(1)
    pred_end = train_end+timedelta(pred_day)

    train = df[train_start:train_end]
    train_diff=train.diff()
    test = df[train_end:]
    test_diff = test.diff()

    print(test_stationarity(train_diff['e'][train_start+timedelta(1):train_end]))
    # 差分后，数据较为平稳  2.2619037813670266e-19
    # plt.figure(facecolor='white',figsize=(20,8))
    # plt.plot(train_diff.index, train_diff['e'],label='Time Series after diff')
    # plt.legend(loc='best')
    # plt.show()
    return index

"""
画出ACF和PACF图
自相关函数ACF，反映了两个点之间的相关性。
偏自相关函数PACF则是排除了两点之间其他点的影响，反应两点之间的相关性。比如：在AR(2)中，即使y(t-3)没有直接出现在模型中，
但是y(t)和y(t-3)之间也相关。
"""

def calculate_ACF(df):

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df['e'][1:], lags=20, ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')

    fig.tight_layout()
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df['e'][1:], lags=20, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    plt.show()


def train_arima(df):
    # 寻找最小的 p,q 值
    pmax,qmax = 8,8
    aic_matrix = []  #aic矩阵
    for p in range(1, pmax+1):
        tmp = []
        for q in range(1, qmax+1):
            model = sm.tsa.arima.ARIMA(endog=df['e'], order=(p, 1, q))
            results = model.fit()
            tmp.append(results.aic)
            print('ARIMA p:{} q:{} - AIC:{}'.format(p, q, results.aic))
        aic_matrix.append(tmp)
    aic_matrix = pd.DataFrame(aic_matrix) #从中可以找出最小值
    p,q = aic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小值位置。
    print(u'AIC最小的p值和q值为：%s、%s' %(p+1,q+1))  # AIC最小的p值和q值为：7、8
    return p+1, q+1


def predict_arima(df, index):
    p,q = 7,8
    model = sm.tsa.arima.ARIMA(endog=df['e'], order=(p,1,q))   #建立ARIMA(7, 1,7)模型
    result_ARIMA = model.fit()

    predict_diff = result_ARIMA.predict()
    #一阶差分还原
    df_shift = df['e'].shift(1)
    predict = predict_diff + df_shift
    df['e'] = df['e'] + df_shift
    # predict = predict_diff
    plt.figure(figsize=(18,5),facecolor='white')

    # predict[train_start+timedelta(p+1):train_end].plot(color='blue', label='Predict')
    # df['e'][train_start+timedelta(p+1):train_end].plot(color='red', label='Original')
    predict[index:].plot(color='blue', label='Predict')
    df['e'][index:].plot(color='red', label='Original')
    # 测试集
    err = sum( np.sqrt((predict[index:]-df['e'][index:])**2)/df['e'][index:]) / df['e'][index:].size
    mse = MSE(predict[index:], df['e'][index:])
    rmse =RMSE(predict[index:], df['e'][index:])

    plt.legend(loc='best')
    plt.show()
    print('error', err*100, ' mes: ', mse, ' rmse: ', rmse)


if __name__ == "__main__":
    df = data_load()
    index = calculate_val(df)
    p,q = train_arima(df)
    predict_arima(df, index)
