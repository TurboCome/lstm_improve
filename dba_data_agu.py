import numpy as np
import random
import math
import sys,os
import pandas as pd

"""
DBA（ DTW Barycentric Averaging）：基于加权形式的 DTW中心平均技术，
通过改变权重，可以从给定的一组时间序列创建无穷多个新的时间序列。作者在3种加权方法中采用了一种叫做平均选择法的加权方法
参考：《Data augmentation using synthetic data for time series classification with deep residual networks》
代码： https://github.com/hfawaz/aaltd18
为使总的权重为1，此时剩余权重为1-0.5-0.15*2=0.2，将剩下的序列平均分配这0.2的权重
"""

""""
1.从训练集中随机选取一个初始时间序列，赋予它0.5的权重，这个随机选择的时间序列将作为DBA的初始化时间序列
2.根据DTW距离，找到DBA初始化时间序列的最近的4 个时间序列，此处是以5种属性为DTW距离，进行的计算。
首先选择最近的2个时间序列，分别赋予0.15 的权重。
然后再找剩余时间序列中的最近的2个序列，分别赋予0.1的权重。
"""

"""
数据量变化： iter = 4
240
480
960
1920
3840

3840 = 
"""


# 计算序列组成单元之间的距离，可以是欧氏距离，也可以是任何其他定义的距离,这里使用绝对值
def distance(w1, w2):
    d = abs(w2 - w1)
    return d

# DTW计算序列s1,s2的最小距离
def DTW(s1, s2):
    m = len(s1)
    n = len(s2)
    # 构建二位dp矩阵,存储对应每个子问题的最小距离
    dp = [[0] * n for _ in range(m)]
    # 起始条件,计算单个字符与一个序列的距离
    for i in range(m):
        dp[i][0] = distance(s1[i], s2[0])
    for j in range(n):
        dp[0][j] = distance(s1[0], s2[j])
    # 利用递推公式,计算每个子问题的最小距离,矩阵最右下角的元素即位最终两个序列的最小值
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + distance(s1[i], s2[j])
    return dp[-1][-1]


def get_four_dis(id, start_id, end_id, df):
    comulns = ['e', 'c', 'h', 'g', 'h2']
    dis_res = []
    list_id = [x for x in df.loc[id].values]
    for i in range(start_id, end_id):
        if i==id: continue
        val = [x for x in df.loc[id].values]
        dis = DTW(list_id[1:], val[1:])
        id_val = [dis, i]
        dis_res.append(id_val)
    dis_sort = sorted(dis_res)
    res = []
    for i in range(4):
        res.append(dis_sort[i][1])
    return res

def dba_data_enhance(df, index_str):

    # df.rename(columns={'date': 'ds', 'Close': 'y'}, inplace=True)
    new_df = df.copy()
    ll = len(df)

    for id in range(0,ll):
        raw = df.loc[id]
        h1,h2,h3,h4 = id,id,id,id
        # if(id>4): start_id = id-5
        # else: start_id = 0
        # if(id<ll-5): end_id = id+5
        # else: end_id = ll-1
        # h1,h2,h3,h4 = get_four_dis(id, start_id, end_id, df)

        if id>1: h1=id-2
        if id>0: h2=id-1
        if id<ll-1: h3=id+1
        if id<ll-2: h4=id+2

        raw1 = df.loc[h1]
        raw2 = df.loc[h2]
        raw3 = df.loc[h3]
        raw4 = df.loc[h4]

        # 0.15,0.15;  0.1,0.1
        rand_arr = [0.15,0.15,0.1,0.1]
        random.shuffle(rand_arr)  # 打乱顺序
        # rdata = raw[1:] *0.5 + raw1[1:] * rand_arr[0] + raw2[1:]*rand_arr[1] + raw3[1:]*rand_arr[2] + raw4[1:]*rand_arr[3]
        # df.loc[id, ('e','c','h','g','h2')] = rdata
        rdata = raw[:] * 0.5 + raw1[:] * rand_arr[0] + raw2[:] * rand_arr[1] + raw3[:] * rand_arr[2] + raw4[:] * rand_arr[3]
        df.loc[id, ('Total#Houses', 'Totallightbulbs', 'Totalgalsgas', 'GHG', 'DOW')] = rdata

        # df.loc[id,'Year'] = str(new_df['Year'].loc[id])
        # new_df.loc[id][1:] = pd.concat([new_df.loc[id],res], axis=1, join='outer')
    return df


def concat_pd(df1, df2, columns):
    ll = len(df1)
    new_df = pd.DataFrame(columns=columns)

    index = 0
    for id in range(0, 2*ll, 2):
        new_df.loc[id] = df1.loc[index]
        new_df.loc[id+1] = df2.loc[index]
        index = index+1
    return new_df.copy()



if __name__:
    # original_path = r"/Users/whq/Public/codeDemo/lstm_improve/data/original.csv"
    original_path = r"/Users/whq/Public/codeDemo/lstm_improve/data/cmcampus_export.csv"
    columns = ['GHG','Total#Houses','Totallightbulbs','Totalgalsgas','DOW']
    # columns = ['KWS', 'KW', 'KW#Houses','KWlightbulbs',  'KWgalsgas']
    # pd.read_csv("somefile.csv", dtype={'column_name': str})
    df = pd.read_csv(original_path, sep=',', usecols=columns, dtype={'data': str})
    # df = pd.read_csv(original_path, sep=',', usecols=columns)
    tmp = df.copy()
    iter = 6
    for i in range(1,iter):
        df1 = dba_data_enhance(tmp.copy(), str(-i))
        df2 = concat_pd(tmp, df1, columns)
        tmp = df2.copy()

    # DBA 数据增强处理后的数据
    sum_df = tmp.copy()
    for id in range(len(sum_df)):
        sum_df.loc[id,'Totallightbulbs'] = sum_df.loc[id,'Totallightbulbs']*0.01

    sum_df.to_csv(r"/Users/whq/Public/codeDemo/lstm_improve/data/dealed_ies_data.csv", index=None)
    #
    # s1 = [1, 3, 2, 4, 2]
    # s2 = [0, 3, 4, 2, 2]
    # print('DTW distance: ', DTW(s1, s2))  # 输出 DTW distance:  2

