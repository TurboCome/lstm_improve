
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

# 解决中文乱码问题
#sans-serif就是无衬线字体，是一种通用字体族。
#常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica, 中文的幼圆、隶书等等。
mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号

def plt_two():
    original_path = r"/Users/whq/Public/codeDemo/lstm_improve/data/original.csv"
    df = pd.read_csv(original_path, encoding='gbk')

    kinds = 'h2'  # e,c,h,g,h2
    x1 = list(range(1,241,1))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel("Original data")
                  # fontweight='bold')
    ax1.plot(x1, df[kinds], c='r')
    ax1.set_xlim(0, 240)

    # plt.plot(x, df[kinds])
    # plt.xlabel("Date", fontsize = 18)
    # plt.ylabel("Normalized value", fontsize = 18)
    # plt.show()

    # plt.figure()
    dealed_path = r"/Users/whq/Public/codeDemo/lstm_improve/data/dealed_dtw_data.csv"
    dealed_df = pd.read_csv(dealed_path, encoding='gbk')
    # x = list(range(1,3841,1))
    # plt.plot(x, df[kinds])
    # plt.show()
    x2 = list(range(1,3841,1))
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel("Enhanced data")
                   # fontweight='bold')
    ax2.plot(x2, dealed_df[kinds], c='g')
    ax2.set_xlim(0, 3840)
    plt.show()


def plt_one():
    # path = r"/Users/whq/Public/codeDemo/lstm_improve/data/IES.csv"
    path = r"/Users/whq/Public/codeDemo/lstm_improve/data/dealed_ies_data.csv"
    # 特征： date,load,year,month,hour,day,lowtmep,hightemp
    # columns =['date','e','c','h','g','ne']
    columns = ['KWS', 'KW', 'KW#Houses','KWlightbulbs',  'KWgalsgas']
    df = pd.read_csv(path, encoding='gbk',sep=',',usecols=columns)

    # df = pd.DataFrame(columns=('data', 'e', 'c', 'h', 'g', 'h2'))
    kinds = 'KWgalsgas'  # date,load,year,month,hour,day,lowtmep,hightemp
    id = round(len(df)*0.3)
    x = list(range(0, id, 1))
    y = df.loc[len(df)-id: len(df), kinds]

    plt.plot(x, y)
    plt.xlabel("Date", fontsize = 18)
    plt.ylabel("Normalized value", fontsize = 18)
    plt.show()

if __name__ == '__main__':
    # plt_two()
    plt_one()