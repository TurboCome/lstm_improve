import matplotlib.pyplot as plt

from dataload import *
from metrics import *
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, SimpleRNN,GRU
import pandas as pd
from keras.layers import Dense, Dropout, Activation,LeakyReLU, concatenate,add
from keras.layers import Convolution2D, MaxPooling2D,AveragePooling2D,GlobalAveragePooling2D
from keras import backend as K
from keras.models import load_model  # 用于加载模型
from keras.utils import plot_model  # 用于可视化
from matplotlib.pyplot import MultipleLocator
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras import optimizers
from tensorflow.keras import Input, Model,Sequential


def scheduler(epoch, lr):
    if epoch % 20 == 0 and epoch != 0 and lr>1e-5:
        return float(lr*0.5)
    return float(lr)


def train_rnn( epochs, batch_size, x_train, y_train, horizon):
    # 初始化顺序模型
    model = Sequential()
    # 循环神经网络,隐藏层100
    model.add(SimpleRNN(units=150, return_sequences=True))
    model.add(Activation('relu'))
    # Dropout层用于防止过拟合
    model.add(Dropout(0.1))
    # 隐藏层100
    model.add(SimpleRNN(units=50))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    # 定义线性的输出层
    model.add(Dense(units=y_train.shape[1],activation='tanh'))
    # 模型编译：定义优化算法adam， 目标函数均方根MSE
    # adam = tf.keras.optimizers.Adam(lr=1e-3)
    # reduce_lr = LearningRateScheduler(scheduler)

    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'acc'])  # adam
    # history = model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size,verbose=1, callbacks=[reduce_lr])
    # model.summary()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    model_name = "rnn_"+str(horizon)
    model.save(model_name)

    # history.history.keys()  # 查看history中存储了哪些参数
    # plt.plot(history.epoch, history.history.get('loss'))  # 画出随着epoch增大loss的变化图
    plot_model(model, to_file='rnn_model.png')  # 将模型model的结构保存到model.png图片


def train_gru(epochs, batch_size, x_train, y_train, horizon):
    model = Sequential()
    # model.add(GRU(units=100,
    #                return_sequences=True,
    #                input_dim=x_train.shape[-1],      #  5
    #                input_length=x_train.shape[1]))   #  120
    # model.add(Dense(units=20,activation='tanh'))
    # # model.add(LeakyReLU(alpha=0.3))
    # model.add(GRU(units=50))
    # model.add(Dense(units=y_train.shape[1],activation='tanh'))
    # model.add(LeakyReLU(alpha=0.3))

    model.add(GRU(units=30,
                  input_dim=x_train.shape[-1],
                  input_length=x_train.shape[1]))
    model.add(Dense(units=y_train.shape[1],activation='tanh'))

    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'acc'])  # adam
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # adam = tf.keras.optimizers.Adam(lr=1e-3)
    # reduce_lr = LearningRateScheduler(scheduler)
    # model.compile(loss='mean_squared_error', optimizer=adam)
    # history = model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[reduce_lr])
    # model.summary()

    model_name = "gru_"+str(horizon)
    model.save( model_name )
    plot_model(model, to_file='gru_model.png')  # 将模型model的结构保存到model.png图片


def train_lstm( epochs, batch_size, x_train, y_train, horizon):
    model = Sequential()
    model.add(LSTM(units=150,
                   return_sequences=True,
                   input_dim=x_train.shape[-1],      #  dim = 5
                   input_length=x_train.shape[1]))   #  length = 120
    model.add(Dense(units=20,activation='tanh'))
    model.add(LSTM(units=50))
    # model.add(Dense(units=20, activation='tanh'))
    # model.add(Dense(units=20, activation='tanh'))
    # model.add(Dropout(0.25))
    model.add(Dense(units=y_train.shape[1],activation='tanh'))

    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'acc'])  # adam
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # adam = tf.keras.optimizers.Adam(lr=1e-3)
    # reduce_lr = LearningRateScheduler(scheduler)
    # model.compile(loss='mean_squared_error', optimizer=adam)
    # history = model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[reduce_lr])
    # model.summary()

    model_name = "lstm_" + str(horizon)
    model.save( model_name )
    plot_model(model, to_file='lstm_model.png')  # 将模型model的结构保存到model.png图片


"""
因为数据增强 将原始数据240，扩大了16倍，到现在 3840；即原始的 1条 数据，相当于现在的连续 16条数据
所以在进行长期预测时，每隔16条数据，即为下一个月的实际预测数据
"""
def train_mst_lstm(epochs, batch_size, x_train, y_train, horizon):
    # LSTM 参数: return_sequences=True LSTM输出为一个序列。默认为False，输出一个值。
    # x_train: (None, 120, 5)
    skip_1 = 2
    skip_2 = 4
    skip_3 = 6
    dims = 5
    input_shape = (1, x_train.shape[1], x_train.shape[2])
    # size of pooling area for max pooling
    pool_size = (1, 1)
    nb_filters = 32  # number of convolutional filters to use

    model1 = Sequential()
    model1.add(Convolution2D(nb_filters,
                             kernel_size=(dims, skip_1),
                             strides=(dims, 1),
                             padding='same',
                             input_shape=input_shape))  # 卷积层1
    model1.add(Activation('relu'))  # 激活层
    model1.add(MaxPooling2D(pool_size))# 池化层
    input1, output1 = model1.input, model1.output  # in(None,1,120,5) ; out(None,1,120,32)

    model2 = Sequential()
    model2.add(Convolution2D(nb_filters,
                             kernel_size=(dims, skip_2),
                             strides=(dims, 1),
                             padding='same',
                             input_shape=input_shape))  # 卷积层1
    model2.add(Activation('relu'))  # 激活层
    model2.add(MaxPooling2D(pool_size))
    input2, output2 = model2.input, model2.output

    model3 = Sequential()
    model3.add(Convolution2D(nb_filters,
                             kernel_size=(dims, skip_3),
                             strides=(dims, 1),
                             padding='same',
                             input_shape=input_shape))  # 卷积层1
    model3.add(Activation('relu'))  # 激活层
    # model3.add(MaxPooling2D(pool_size))
    input3, output3 = model3.input, model3.output  #(None, 1,  120, 32)

    # added = add([output1*0.2, output2*0.2, output3*0.6]) # (None,1,120,32)
    concatenated = concatenate([output1*0.2, output2*0.2, output3*0.6], axis=-1)  # (None,1,120,96)

    fc1 = Dense(units=48, input_dim = nb_filters*3, activation='tanh')(concatenated)
    # fc2 = Dense(units=nb_filters*3, input_dim=round(nb_filters*3/2), activation='tanh')(fc1)
    pool_fc = MaxPooling2D(pool_size)(fc1)
    concatenated_squeeze = K.squeeze(pool_fc, 1)

    lstm_out1 = LSTM(units=200,
         return_sequences=True,
         input_dim = 48,
         input_length = 120)(concatenated_squeeze)
    dense_out = Dense(units=30, activation='tanh')(lstm_out1)
    lstm_out2 = LSTM(units=60)(dense_out)  #(None, 120, 100)
    # dense1 = Dense(units=20, activation='tanh')(lstm_out2)
    # dense2 = Dense(units=20, activation='tanh')(dense1)
    # dropout = Dropout(0.25)(dense2)
    dese_output = Dense(units=y_train.shape[1], activation='tanh')(lstm_out2)

    model = Model(inputs=[input1, input2, input3], outputs=dese_output)

    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae', 'acc'])  # adam
    input = np.expand_dims(x_train, axis=1)
    history = model.fit([input,input,input], y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # adam = tf.keras.optimizers.Adam(lr=0.005)
    # reduce_lr = LearningRateScheduler(scheduler)
    # model.compile(loss='mean_squared_error', optimizer=adam)
    # history = model.fit([input,input,input], y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[reduce_lr])
    # model.summary()

    model_name = "mst_lstm_"+str(horizon)
    model.save(model_name)
    history.history.keys()  # 查看history中存储了哪些参数
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.xlabel("epochs ",fontsize=10)
    plt.ylabel("train loss ",fontsize=10)
    plt.plot(history.epoch, history.history.get('loss'))  # 画出随着epoch增大loss的变化图
    # plot_model(model, to_file='mst_lstm_model.png') #将模型model的结构保存到model.png图片



def predict( model_name, x_test, y_test, scaler, horizon):
    # model_path = r"/Users/whq/Public/codeDemo/lstm_improve/guojiatongji/" + str(model_name)+"_"+str(horizon)

    model_path = r"/Users/whq/Public/codeDemo/lstm_improve/" + str(model_name) + "_" + str(horizon)
    model = load_model(model_path)
    if model_name =='mst_lstm':
        input = np.expand_dims(x_test,axis=1)
        y_pred = model.predict([input,input,input])
    else: y_pred = model.predict(x_test)
    # MAE(pred, true) MSE(pred, true)  RMSE(pred, true)  MAPE(pred, true) MSPE(pred, true)
    print('model:', model_name, 'mse:', MSE(y_pred,y_test), ' rmse:', RMSE(y_pred,y_test),
          'mae:', MAE(y_pred,y_test), 'rse:', RSE(y_pred, y_test), 'corr:', CORR(y_pred, y_test),'mape:',MAPE(y_pred, y_test)) # 'mape:',MAPE(pred, true),'mspe:',MSPE(pred, true) )

    # 反归一化  source_x_test: (588, 5);  predict:(588,60)
    source_x_test = x_test[:, 0, 1:]
    predict, y_test_single = y_pred[:, 0], y_test[:, 0]
    pred_list = scaler.inverse_transform(np.column_stack((source_x_test, predict)))
    true_list = scaler.inverse_transform(np.column_stack((source_x_test, y_test_single)))
    pred, true  = pred_list[:, -1],true_list[:, -1]

    # pred,true = y_pred[:, 0],y_test[:, 0]
    # 预测60个月的数据
    # source_x_test = x_test[:60, 0, 1:]
    # predict, y_test_single = y_pred[0, 0:60].transpose(), y_test[0, 0:60].transpose()


    # pred, true = pred.reshape(1,-1)[0], true.reshape(1,-1)[0]
    # pred_s,true_s = [],[]
    # for i in range(0,len(pred),horizon):
    #     pred_s.append(pred[i])
    #     true_s.append(true[i])

    # 对比图
    # plt.figure(figsize=(16, 8))
    # dict_data = {'Predictions': pred,'e': true}
    # data_pd = pd.DataFrame(dict_data)
    # plt.plot(data_pd[['e', 'Predictions']])
    # plt.show()
    return pred, true


def plot_models(rnn, lstm, gru, mst_lstm, true):

    plt.figure(figsize=(16, 8))
    # dict_data = {'Predictions': pred_s,'e': true_s}
    # data_pd = pd.DataFrame(dict_data)

    # plt.rcParams['font.sans-serif'] = [u'SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    markerSize = 1.5
    lineWidth = 1.5
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

    x = list(range(0,len(true),1))
    # plt.plot(data_pd[['e', 'Predictions']])
    plt.plot(x, true, color='#D3D3D3', linestyle='-', linewidth=lineWidth, label="真实值")
    plt.plot(x, mst_lstm, color='#DC143C', linestyle='-', linewidth=lineWidth, label="MST-LSTM")
    plt.plot(x, gru, color='#0000FF', linestyle='--', linewidth=lineWidth, label="GRU")
    plt.plot(x, lstm, color='#228B22', linestyle='-.', linewidth=lineWidth, label="LSTM")
    plt.plot(x, rnn, color='#FFA500', linestyle='-.', linewidth=lineWidth, label="RNN")

    grid_x,grid_y = 50, 0.2
    xUp = 800
    yDown,yUp = 7,8.2
    x_major_locator = MultipleLocator(grid_x)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(grid_y)      # mse 0.5;  rse:
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(0, xUp)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(yDown, yUp)    # rse: ydown=20
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

    plt.legend()
    plt.grid(linestyle=':')
    plt.xlabel("时间/月",fontsize=10)
    # r'(' + '$\mu\mathrm{mol}$' + ' ' + '$m^{-2}.s^{-1})$' + ' $A_n$',
    plt.ylabel("电负荷/10^3 KW",fontsize=10)
    plt.show()
    plt.savefig('4种算法对比图')



if __name__ == "__main__":
    # 加载数据
    # data_path = r"/Users/whq/Public/codeDemo/lstm_improve/data/dealed_data.csv"
    # columns = ['e', 'c', 'h', 'g', 'h2']

    # data_path = r"/Users/whq/Public/codeDemo/lstm_improve/data/IES.csv"
    # columns = ['e','c','h','g','ne']

    data_path = r"/Users/whq/Public/codeDemo/lstm_improve/data/dealed_ies_data.csv"
    columns = ['Total#Houses','Totalgalsgas','Totallightbulbs','GHG','DOW']

    df = load_data(data_path,columns)

    # for horizon_init in range(12,73,12):
    # 超参数
    horizon_init = 6
    window = round(horizon_init*2)
    horizon = horizon_init
    epochs = 15
    batch_size = 64
    idx = 0    # ['e','c','h','g','ne']
    x_train, y_train, x_test, y_test, scaler = custom_ts_multi_data_prep(df, window, horizon, idx)

    #预测、预测模型
    train_rnn(epochs, batch_size, x_train, y_train,horizon)
    rnn, true = predict( 'rnn', x_test, y_test, scaler,horizon)

    train_lstm(epochs, batch_size, x_train, y_train, horizon)
    lstm, true = predict( 'lstm', x_test, y_test, scaler,horizon)
    #
    train_gru(epochs, batch_size, x_train, y_train, horizon)
    gru, true = predict('gru', x_test, y_test, scaler,horizon)

    train_mst_lstm(epochs, batch_size, x_train, y_train, horizon)
    mst_lstm, true = predict('mst_lstm', x_test, y_test, scaler, horizon)

    # 4种算法画图
    plot_models(rnn,lstm,gru,mst_lstm,true)

