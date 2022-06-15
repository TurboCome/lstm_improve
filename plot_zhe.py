import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator


def deal_y(list_y, a):
    res = []
    for v in list_y:
        res.append(v*a)
    return res

def plot_three():
    x = [ 12,24,36,48,60,72]

    # MSE:
    rnn_mse =[0.007027,0.013747,0.018793,0.021830,0.028248,0.029030]
    lstm_mse=[0.005175,0.008124,0.013409,0.021096,0.026247,0.029821]
    gru_mse= [0.004096,0.009265,0.014629,0.018501,0.024502,0.030220]
    mst_lstm_mse= [0.004056,0.007691,0.013259,0.016687,0.022636,0.027389]

    # RSE
    rnn_rse=[37.77,51.82,60.99,66.62,78.41,87.51]
    lstm_rse=[32.41,39.83,51.51,65.49,75.58,88.70]
    gru_rse=[28.83,42.54,53.81,61.33,73.02,89.29]
    mst_lstm_rse=[28.69,38.76,51.90,56.83,70.19,85.00]

    # CORR:
    rnn_corr=[0.077904,0.035891,0.022196,0.015782,0.010549,0.007185]
    lstm_corr=[0.079872,0.038515,0.025008,0.016285,0.011201,0.007012]
    gru_corr=[0.079950,0.037950,0.023709,0.016531,0.011585,0.007856]
    mst_lstm_corr=[0.079957,0.038729,0.023757,0.016816,0.011886,0.007829]

    # MAE
    rnn_mae=[0.066453, 0.094827,0.107635,0.11767,0.131365,0.132861]
    lstm_mae=[0.052035,0.065694,0.085321,0.10703,0.122418,0.134483]
    gru_mae=[0.048500,0.073278,0.090688,0.09987,0.116517,0.123330]
    mst_lstm_mae=[0.048048,0.062831,0.090688,0.09078,0.118243,0.124792]

    a = 100
    y0 = deal_y(rnn_mae,a)
    y1 = deal_y(lstm_mae ,a)
    y2 = deal_y(gru_mae ,a)
    y3 = deal_y(mst_lstm_mae ,a)



    yName = 'MAE(%)'
    xName = '预测时间序列长度(月)'
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    markerSize = 4
    lineWidth = 1.5
    plt.plot(x, y0, color='#FFA500', marker='+', linestyle='-.', linewidth=lineWidth, label="RNN",markersize=markerSize)
    plt.plot(x, y1, color='#228B22', marker='v', linestyle='-.', linewidth=lineWidth, label="LSTM",markersize=markerSize)
    plt.plot(x, y2, color='#0000FF', marker='*', linestyle='-.', linewidth=lineWidth, label="GRU", markersize=markerSize)
    plt.plot(x, y3, color='#DC143C', marker='o', linestyle='-', linewidth=lineWidth, label="MST-LSTM",markersize=markerSize)
    # plt.plot(x, y4, color='#FF00FF', marker='<', linestyle='--', linewidth=lineWidth, label="MCC", markersize=markerSize)
    # plt.plot(x, y5, color='#4B0082', marker='^', linestyle='-.', linewidth=lineWidth, label="FTA",markersize=markerSize)
    # plt.plot(x, y6, color='#A2142F', marker='o', linestyle='-', linewidth=lineWidth, label="FTA+DTA",markersize=markerSize)
    xUp = 75
    yUp = 15  #rse
    grid_x = 12
    grid_y = 3
    plt.xlabel(xName, fontsize=10)
    plt.ylabel(yName, fontsize=10)

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
    plt.ylim(0, yUp)    # rse: ydown=20
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.legend()
    plt.grid(linestyle='-.')
    plt.show()


def plot_zhu2():

    name = ["电", "冷", "热", "气", "新能源"]
    rnn_rmse = [0.1000, 0.0675, 0.1069, 0.0997, 0.0798]
    lstm_rmse = [0.0720, 0.0505, 0.0615, 0.0652, 0.0708]
    gru_rmse = [0.0757, 0.0510, 0.0779, 0.0774, 0.0802]
    mst_lstm_rmse = [0.0577, 0.0460, 0.0577, 0.0644, 0.0671]

    rnn_mape = [25.0710, 9.1618, 29.3093, 42.6474, 46.1174]
    lstm_mape = [19.1013, 6.3124, 12.8650, 28.2050, 41.9475]
    gru_mape = [15.0840, 6.3572, 15.2220, 35.3081, 40.6536]
    mst_lstm_mape = [12.1140, 6.0427, 12.1140, 35.8374, 31.3690]

    a = 1
    y0 = deal_y(rnn_mape, a)
    y1 = deal_y(lstm_mape, a)
    y2 = deal_y(gru_mape, a)
    y3 = deal_y(mst_lstm_mape, a)

    x = np.arange(len(name))
    width = 0.12

    plt.bar(x+width, y0, width=width, label='RNN', color='#FFA500')
    plt.bar(x + width*2, y1, width=width, label='LSTM', color='#228B22', tick_label=name)
    plt.bar(x + width*3, y2, width=width, label='GRU', color='#0000FF')
    plt.bar(x + width*4, y3, width=width, label='MST-LSTM', color='#DC143C')

    # 显示在图形上的值
    # for a, b in zip(x, y0):
    #     plt.text(a, b + 0.1, b, ha='center', va='bottom')
    # for a, b in zip(x, y1):
    #     plt.text(a + width, b + 0.1, b, ha='center', va='bottom')
    # for a, b in zip(x, y2):
    #     plt.text(a + 2 * width, b + 0.1, b, ha='center', va='bottom')
    # for a, b in zip(x, y3):
    #     plt.text(a + 3 * width, b + 0.1, b, ha='center', va='bottom')

    xName = '综合能源'
    yName = 'MAPE(%)'
    yUp = 50
    plt.xticks()
    plt.grid(axis="y", linestyle='-.')

    # plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
    plt.legend()
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.ylabel(yName)
    plt.xlabel(xName)
    plt.yticks(np.arange(0, yUp, 5))
    # plt.rcParams['savefig.dpi'] = 300  # 图片像素
    # plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 尺寸
    plt.title("title")
    plt.savefig('rmse.png')
    plt.show()


def plot_Zhu():
    x = [ 12,24,36,48,60,72]

    # RMSE: 电，冷，热，气，新能源
    rnn_rmse = [0.100099, 0.067508  , 0.106946 ,0.0997, 0.079870]
    lstm_rmse = [0.072093, 0.050572, 0.061575, 0.0652, 0.070883]
    gru_rmse = [0.075769, 0.051097,  0.077906, 0.0774, 0.080215 ]
    mst_lstm_rmse= [0.057764, 0.046015, 0.057764,0.0644,0.067168 ]

    rnn_mape = [25.0710,9.1618,29.3093,72.6474,46.1174]
    lstm_mape = [19.1013 ,6.3124,12.8650,28.2050,41.9475]
    gru_mape = [15.0840 ,6.3572,15.2220,35.3081,40.6536]
    mst_lstm_mape = [12.1140 ,6.0427,12.1140,35.8374,31.3690]


    a = 100
    y0 = deal_y(rnn_rmse,a)
    y1 = deal_y(lstm_rmse ,a)
    y2 = deal_y(gru_rmse,a)
    y3 = deal_y(mst_lstm_rmse ,a)

    bar_width = 0.15  # 0.2 # 条i+1
    xName = u'任务数量（个）'
    yName = u'分区窗口占用率（%）'
    xUp = 200
    yUp = 100

    ll = int(len(y1))
    index_init = np.arange(ll)
    index1 = index_init + bar_width * 1
    index2 = index_init + bar_width * 2
    index3 = index_init + bar_width * 3
    index4 = index_init + bar_width * 4
    index5 = index_init + bar_width * 5
    # index_LSTM = index_LIF + 3 * bar_width
    # 防止乱码
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    # plt.rcParams['font.sans-serif'] = ['SimHei'] # 如果要显示中文字体,则在此处设为：SimHei
    # plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    # plt.title("Fault detection accuracy")

    # 设置y轴刻度，使其以10的幂次方来显示
    # plt.set_ylim(10**-2, 10**2)
    # plt.set_yscale('log')

    #  A3C > TLG  > CMA  >  FTA >  FTADTA
    plt.legend()
    # plt.grid(linestyle='-.')
    plt.grid(axis="y", linestyle='-.')
    plt.bar(index_init, height=y0, width=bar_width, color='#EE82EE', label='DRA')
    plt.bar(index1, height=y1, width=bar_width, color='#77AC30', label='A3C')
    plt.bar(index2, height=y2, width=bar_width, color='#D95319', label='TLG')
    plt.bar(index3, height=y3, width=bar_width, color='#0072BD', label='CMA')
    # plt.bar(index4, height=y4, width=bar_width, color='#4B0082', label='FTA')
    # plt.bar(index5, height=y5, width=bar_width, color='#A2142F', label='FTA+DTA')

    plt.xticks(index_init + 1.5 * bar_width, x)
    # #控制y轴的刻度
    plt.yticks(np.arange(0, yUp, 10))

    font = FontProperties(size=8)  # “”里面为字体的相对地址 或者绝对地址
    plt.xlabel(xName, fontproperties=font)
    plt.ylabel(yName, fontproperties=font)
    plt.legend()
    plt.savefig("recv2", dpi=200)
    plt.show()



if __name__:
    # plot_three()
    plot_zhu2()
