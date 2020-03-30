import pandas as pd
import numpy as np
from numpy import nan
from sklearn import preprocessing

#加载均值
data = pd.read_csv('./data/患者均值中值.csv', encoding='utf-8',low_memory=False)
data_mean = data.iloc[0,range(0,35)] #36个指标的均值

#加载指标表
table = pd.read_csv('./data/患者.csv', encoding='utf-8',low_memory=False)
table1 = table.iloc[range(0,4788),range(0,35)] #4788是行数，从第7列开始取，到第36列
for j in range(0,35):
    for i in range(len(table1.iloc[:,0])):
        if np.isnan(table1.iloc[i, j]):
            table1.iloc[i, j] = data_mean[j]

table1.to_csv("./data/患者均值插值结果.csv",index=False,sep=',', encoding='utf-8')