# encoding=utf-8
# @FileName  :gaussNoise.py
# @Time      :2022-02-19 11:16
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
import sys

sys.path.append("./")
if __name__ == '__main__':
    origin = [0,0]
    popu= 100
    sigma ,u= 2.5,0
    addNosie = []
    x_list=[]
    y_list=[]

    # 绘图范围
    xmin,xmax= -9,9
    ymin,ymax= -9,9
    for _ in range(popu):
        noise =sigma*np.random.randn( len(origin ) ) +u
        x_list.append(noise[0]+origin[0])
        y_list.append(noise[0+1]+origin[0+1])
    # 解决不能显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("")
    plt.xlim(xmax=xmax, xmin=xmin)
    plt.ylim(ymax=ymax, ymin=ymin)
    # 点的坐标分别为[0, 0], [3, 8], [1, 1], [2, 2], [5, 3], [4, 8], [6, 3], [5, 4], [6, 4], [7, 5]
    plt.plot(x_list,y_list, 'ro',)
    plt.savefig('testblueline.jpg')
