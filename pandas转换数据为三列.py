矩阵转换为xyz三列数据
# encoding=utf-8
# @FileName  :pandas转换数据为三列.py
# @Time      :2022-01-04 17:00
# @Author    :wxt
import numpy as np
from optparse import OptionParser
import random
import pickle
import pprint
import logging
import os
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import h5py
import pandas as pd



def savenpyasexcel(ndarray,output):
    #ndarray是数组，output是保存的文件位置
    import pandas as pd
    data_df = pd.DataFrame(ndarray)  # 关键1，将ndarray格式转换为DataFrame
    # rows,cols = ndarray.shape
    # # 更改表的索引
    # data_index = []
    # for i in range(rows):
    #     data_index.append(i)
    # data_df.index = data_index
    # # 更改表的索引
    # data_index = []
    # for i in range(cols):
    #     data_index.append(i)
    # data_df.index = data_index
    # data_df.columns = data_index

    # 将文件写入excel表格中
    writer = pd.ExcelWriter(output)  # 关键2，创建名称为hhh的excel表格
    data_df.to_excel(writer, 'page_1',
                     float_format='%.55f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    writer.save()  # 关键4
    return 1
if __name__ == '__main__':
    data = r'F:\wxt\每周周报\计算的超振荡20211127发送' \
           r'\发送给zz的设置轮廓图和3d图2021-12-20画图使用\1d转换原始数据.xlsx'
    data =pd.read_excel(data,header=None)
    print(data)
    data = np.array(data)
    print(data)
    print(data.shape)
    # 100 , 200
    finalRes = np.zeros((data.shape[0] * data.shape[1],3),dtype=float)
    # 100*100,3
    rowCount = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            tempRow = rowCount
            rowCount+=1
            tempx1,tempx2 = i,j
            tempVal = data[i,j]
            threshold = 1.7e14
            if tempVal > threshold:
                tempVal = threshold
            if tempVal < -threshold:
                tempVal = -threshold

            finalRes[tempRow,0],finalRes[tempRow,1],finalRes[tempRow,2] \
                = tempx1,tempx2,tempVal
            print(tempRow)
    # print(finalRes)
    print(finalRes.shape)
    savenpyasexcel(finalRes,"temp2d转换原始截面.xlsx")
