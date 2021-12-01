# encoding=utf-8
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
class Logger(object):

    def __init__(self, log_file_name, log_level, logger_name):
        # 创建一个logger
        self.__logger = logging.getLogger(logger_name)

        # 指定日志的最低输出级别，默认为WARN级别
        self.__logger.setLevel(log_level)

        # 创建一个handler用于写入日志文件
        file_handler = logging.FileHandler(log_file_name)

        # 创建一个handler用于输出控制台
        console_handler = logging.StreamHandler()

        # 定义handler的输出格式
        formatter = logging.Formatter(
            '[%(asctime)s] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 给logger添加handler
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def trans_realK_to_complexK(k_real):
    # k_real is 80,1的数据
    # trans a real array to a complex array
    import numpy as np
    k_complex = np.zeros(shape=(len(k_real)//2,1),dtype=complex)
    for i in range(k_complex.shape[0]):
        k_complex[i,0] = k_real[i] +  (1j * k_real[i+k_complex.shape[0]])
    return k_complex.tolist()

def getIntensity(array, Firstdim=1):
    a,fengliang,points = array.shape
    res = np.zeros(shape=(1,points),dtype=float)
    for j in range(points):
        for i in range(fengliang):
            res[0,j] += np.abs(array[i,j])**2
    return res

def getIntensity(array):
    if 1==array.shape[0]:
        return getIntensity(array =array,Firstdim=1)
    else:
        fengliang, points = array.shape
        res = np.zeros(shape=(1, points), dtype=float)
        for j in range(points):
            for i in range(fengliang):
                res[0, j] += np.abs(array[i, j]) ** 2
        return res

# def kcomplex_X_originmodelist(k_complex,originModeList):
#     datares = np.zeros(shape=originModeList[0].shape, dtype=originModeList[0].dtype)
#     for modeNumber in range(len(originModeList)):
#         occupied = originModeList[modeNumber] * k_complex[modeNumber]
#         datares += occupied
#     return datares

def kcomplex_X_originmodelist(k_complex,originModeList):
    print(originModeList[0].shape)
    datares = torch.zeros((originModeList[0].shape[0],originModeList[0].shape[0+1]),dtype=torch.complex64)
    for modeNumber in range(len(originModeList)):
        occupied = originModeList[modeNumber] * k_complex[modeNumber]
        datares += occupied
    return datares

def getNewKcomplex(kcomplex ,deitaAngle):
    r,angle = getR_and_angle(kcomplex)
    complex01 = r * np.cos(angle + deitaAngle) + r * np.sin(angle + deitaAngle) * 1j
    return complex01

def distanceAFromB(dataA,dataB):
    return np.sum((dataA-dataB)**2)


def readMat(matFile='',readdata_dict="xxex"):
    import h5py
    path = matFile  # 需要读取的mat文件路径
    data = h5py.File(path)  # 读取mat文件
    ex = np.array(data[readdata_dict])
    ex =  np.transpose(ex,(1,0))
    return ex

def readmatUsingScio(matfile,readdata_dict  ):
    import scipy.io as scio  
    dataall = scio.loadmat(matfile)
    ex = dataall[readdata_dict]
    ex = np.transpose(ex, (2,1, 0))
    return ex

def readmatUsingSciokkk(matfile,readdata_dict  ):
    import scipy.io as scio
    dataall = scio.loadmat(matfile)
    ex = dataall[readdata_dict]
    ex = np.transpose(ex, (1, 0))
    return ex

def splitNumpyAsBlock(data,blockSize):
    blockList =[]
    rows = data.shape[0]
    cols = data.shape[1]

    data = data
    blockNumberrows = rows // blockSize
    blockNumbercols = cols // blockSize
    data = data[:blockSize * blockNumberrows, :blockSize * blockNumbercols]

    for i in range(blockNumberrows):
        indexTouseRowStart = i*blockSize
        indexTouseRowStop = i*blockSize + blockSize
        for j in range(blockNumbercols):
            indexTouseColStart  = j * blockSize
            indexTouseColStop  = j * blockSize + blockSize
            blockList.append( data[indexTouseRowStart:indexTouseRowStop
                              ,indexTouseColStart:indexTouseColStop] )

    return blockList


def splitNumpyAsBlock(data,blockSize,getMean=True):
    blockListMean = []
    rows = data.shape[0]
    cols = data.shape[1]

    data = data
    blockNumberrows = rows // blockSize
    blockNumbercols = cols // blockSize
    data = data[:blockSize * blockNumberrows, :blockSize * blockNumbercols]

    for i in range(blockNumberrows):
        indexTouseRowStart = i*blockSize
        indexTouseRowStop = i*blockSize + blockSize
        for j in range(blockNumbercols):
            indexTouseColStart  = j * blockSize
            indexTouseColStop  = j * blockSize + blockSize
            # blockList.append( data[indexTouseRowStart:indexTouseRowStop
            #                   ,indexTouseColStart:indexTouseColStop] )
            blockListMean.append(np.mean( 
                data[indexTouseRowStart:indexTouseRowStop
                                  ,indexTouseColStart:indexTouseColStop]))
    return np.array(blockListMean)

def getRandomKcomplex(kComplexNumber, rangeUsed=1):
    preserKcomlexList = []

    for _ in range(kComplexNumber):
        areal = np.random.rand() * rangeUsed *2- rangeUsed
        aimage = np.random.rand() * rangeUsed *2- rangeUsed
        acomplex = areal + 1j * aimage
        preserKcomlexList.append(acomplex)
    return preserKcomlexList


def getRandomKcomplexBIggerThan0(kComplexNumber, rangeUsed=1):
    preserKcomlexList = []
    for _ in range(kComplexNumber):
        areal = np.random.rand() * rangeUsed
        aimage = np.random.rand() * rangeUsed
        acomplex = areal + 1j * aimage
        preserKcomlexList.append(acomplex)
    return preserKcomlexList
   


def makedir(dir):
    import os
    dir = dir.strip()
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        return True
    else:
        return False

def savenpyasexcel(ndarray,output):
	#ndarray是数组，output是保存的文件位置
    import pandas as pd
    data_df = pd.DataFrame(ndarray)  # 关键1，将ndarray格式转换为DataFrame
    rows,cols = ndarray.shape
    # 更改表的索引
    data_index = []
    for i in range(rows):
        data_index.append(i)
    data_df.index = data_index
    data_df.index = data_index
    # 更改表的索引
    data_index = []
    for i in range(cols):
        data_index.append(i)
    data_df.columns = data_index

    # 将文件写入excel表格中
    writer = pd.ExcelWriter(output)  # 关键2，创建名称为hhh的excel表格
    data_df.to_excel(writer, 'page_1',
                     float_format='%.25f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    writer.save()  # 关键4
    return 1

def makeListToPsoParticles(kList, writer="Wangxutao"):
    res_list = []
    for i in range(len(kList)):
        res_list.append(np.real(kList[i]))


    for i in range(len(kList)):
        res_list.append(np.imag(kList[i]))
    return  res_list

def getOriginKcomplex_add_noise(arrList,populations = 120):
    arrList_complex_real_image = makeListToPsoParticles(arrList)

    resList =[]
    for i in range(populations):
        arrListCopy = arrList_complex_real_image.copy()
        # tempNoise = np.random.randn(3,2) # 3行2列的高斯矩阵
        tempNoise = np.random.randn(len(arrList_complex_real_image))
        tempNoise = tempNoise.tolist()
        for data in range(len(arrListCopy)):
            arrListCopy[data] = arrListCopy[data] +tempNoise[data]
        resList.append(arrListCopy)
    return  resList


def saveNewMadeData(dataList , savePlace):
    return pickcleSaveData(dataList,savePlace)

def pickcleSaveData(data,save_place):
    data = data
    output = open(save_place, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(data, output)
    output.close()
    return "success save!"

def savePkl(data,save_place):
    return pickcleSaveData(data,save_place)

def readDataFromFile(filename):
    pkl_file = open(filename, 'rb')
    data1 = pickle.load(pkl_file)
    # pprint.pprint("readdata ",data1)
    pkl_file.close()
    return data1
def loadPkl(filename):
    return readDataFromFile(filename)

def getAllFilesFromDir(fileDir):
    listHere = os.listdir(fileDir)
    return listHere



def resizeFrom1601to801(dataArray,from_size=1601,tosize=801):

    # 1,1601, 1601
    data = dataArray
    if(data.shape[1]==tosize):
        return data,False
    print("from  ",data.shape)
    data = np.reshape(data , (from_size,from_size))

    print("to    ", data.shape)
    # save_pic(data, save_place=r"C:\psoInverseProblem\深度学习的与训练模型生成\savePlaceMode01andmode03\data_saveDir\3c66be2d0b864adba4c0b696e5394b9a.png_data.png".replace(".png","copy.png")
    #          ,shape =1601)

    # exit(0)

    rows,cols = data.shape[0], data.shape[1]
    midIndex = cols // 2 + 1

    data_getMiddle = data[midIndex  - tosize// 2:midIndex  + tosize// 2 + 1,
                     midIndex - tosize // 2:midIndex + tosize//2+1]

    data = np.reshape(data_getMiddle, (1,tosize,tosize))

    # save_pic(data,
    #          save_place=r"C:\psoInverseProblem\深度学习的与训练模型生成\savePlaceMode01andmode03\data_saveDir\3c66be2d0b864adba4c0b696e5394b9a.png_data.png".replace(
    #              ".png", "801copy.png")
    #          , shape=801)
    # exit(0)
    return data,True

def scale_array(src, new_size):
    res = cv2.resize(src,new_size)
    return res

def resizeFromxxto41(dataArray , fromsize =(1,801,801),tosize=(41,41)):

    # no need resize
    if(dataArray.shape==tosize):
        return dataArray,False
    shape0,shape1,shape2  = dataArray.shape
    dataArray = np.reshape(dataArray,(shape1,shape2))

    res=scale_array(dataArray,tosize)
    res = np.reshape(res,(1,res.shape[0],res.shape[1]))
    return  res,True

def resizeFrom801to41(dataArray ,tosize=(41,41)):

    # no need resize
    if(dataArray.shape==tosize):
        return dataArray,False

    dataArray = np.reshape(dataArray,(801,801))
    res=scale_array(dataArray,tosize)
    res = np.reshape(res,(1,res.shape[0],res.shape[1]))
    return  res,True

def transDataFromDoubleToFloat32(dataArray):
    data = dataArray
    typeHere =data.dtype
    if(data.dtype=="float32"):
        return data,False

    data = data.astype(np.float32)
    newtypehere = data.dtype


    return data,True

def readLabelFromFile(labelFileSavePlace):
    label = readDataFromFile(labelFileSavePlace)
    label_first_mode = label[0]
    label_second_mode = label[0 + 1]
    label_1mode_real = np.real(label_first_mode)
    label_1mode_imag = np.imag(label_first_mode)
    label_2mode_real = np.real(label_second_mode)
    label_2mode_imag = np.imag(label_second_mode)

    # data = data.astype(np.float32)
    # data.to(torch.float32)
    # label = label_1mode_real
    label = np.array([label_1mode_real, label_2mode_real, label_1mode_imag, label_2mode_imag])
    return  label




def options_func():
    optParser = OptionParser()

    optParser.add_option('--input_dim', action="store", type="int", dest='input_dim', default=801*801,
                         help="Dimension of train data.")
    optParser.add_option('--output_dim', action="store", type="int", dest='output_dim', default=4,
                         help="Dimension of output.")

    optParser.add_option('--k1', action="store", type="int", dest='k1', default=5, help="The count of hidden nodes.")
    optParser.add_option('--k2', action="store", type="int", dest='k2', default=5, help="The count of hidden nodes.")
    optParser.add_option('--k3', action="store", type="int", dest='k3', default=5, help="The count of hidden nodes.")
     
    optParser.add_option("--dataset", action="store", type="str", dest='dataset', default='random',
                         help="Choosing which dataset to use.")
    optParser.add_option("--model", action="store", type="str", dest='model', default='LL',
                         help="Choosing which net structure to use.")
    optParser.add_option('--lr', action="store", type="float", dest='lr', default=0.01,
                         help="The learning rate of training.")

    optParser.add_option("--train_epoch", action="store", type="int", dest='train_epoch', default=200,
                         help="The count of training epoch.")
    optParser.add_option('--sample_rate', action="store", type="float", dest='sample_rate', default=1,
                         help="The rate of sampling.")

    optParser.add_option('--pre_train', action="store_true", dest='pre_train', default=False,
                         help="Pre_train or not.")
    optParser.add_option('--norm_freq', action="store", type="int", dest='norm_freq', default=5,
                         help="The freq of norm.")
    optParser.add_option('--norm', action="store_true", dest='norm', default=True, help="Normalization or not.")

    optParser.add_option('--batch_size', action="store", type="int", dest='batch_size', default=20000,
                         help="The batch of test data.")

    options, arguments = optParser.parse_args()
    print('\n------------ Options -------------\n', options, '\n-------------- End ----------------\n')

    return options, arguments

def getOriginDataFromNormation(data =0,min =-4 ,max=4,norRange= 1):
    data = norRange -data
    data = data * (max - min)
    return min + data

def resizeLable(file):
    data = loadPkl(file)
    data = data
    return  data

def getIntensityOfMode(mode):
    b,c= mode.shape
    print(mode.shape)
    jingdu =(int)(c**0.5)

    tempez = np.reshape(mode[0,:],  (jingdu,jingdu))
    tempex = np.reshape(mode[1,:],  (jingdu,jingdu))
    tempey = np.reshape(mode[2,:],  (jingdu,jingdu))

    res = (np.abs(tempex))**2+(np.abs(tempey))**2+(np.abs(tempez))**2
    print(res.shape)
    return res

def normDatafrom0to1(data,toRange=1.0):
    data_MAX= torch.max(data)
    data_MIN= torch.min(data)
    data_range = data_MAX-data_MIN
    data = (data-data_MIN)/(data_range) * toRange
    return data



def save_pic(arr,save_place,shape=801,title='little',style='jet'):
    plt.ion()
    # array = np.zeros(shape=arr.shape,dtype=np.int64)
    # for  i in range(arr.shape[0]):
    #     for j in range(arr.shape[1]):
    #         if arr[i,j]>0:array[i,j]=255
    # print(array)
    arr = np.reshape(arr,(shape,shape))
    esum_temp =arr
    plt.imshow(esum_temp, cmap=style)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('intensity', rotation=-90, va='bottom')

    max_index = np.max(esum_temp)
    min_index = np.min(esum_temp)
    interval_temp = (max_index - min_index) / 5
    cbar.set_ticks([min_index, min_index + 1 * interval_temp,
                    min_index + 2 * interval_temp, min_index + 3 * interval_temp,
                    min_index + 4 * interval_temp, min_index + 5 * interval_temp])
    # set the font size of colorbar
    cbar.ax.tick_params(labelsize=8)
    save_place = save_place
    plt.savefig(save_place)
    plt.pause(0.005)
    plt.close('all')
    return esum_temp

def copyFileDirFromTo(dir_from,dir_to):

    makedir(dir_to)
    list_here = os.listdir(dir_from)
    return 0



mport numpy as np

import cv2
import copy
def scale_array(src, new_size):
    res = cv2.resize(src,new_size)
    return res

def savenpyasexcel(ndarray,output):
	#ndarray是数组，output是保存的文件位置
    import pandas as pd
    data_df = pd.DataFrame(ndarray)  # 关键1，将ndarray格式转换为DataFrame
    rows,cols = ndarray.shape
    # 更改表的索引
    data_index = []
    for i in range(rows):
        data_index.append(i)
    data_df.index = data_index
    data_df.index = data_index
    # 更改表的索引
    data_index = []
    for i in range(cols):
        data_index.append(i)
    data_df.columns = data_index

    # 将文件写入excel表格中
    writer = pd.ExcelWriter(output)  # 关键2，创建名称为hhh的excel表格
    data_df.to_excel(writer, 'page_1',
                     float_format='%.25f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    writer.save()  # 关键4
    return 1




# 再一个矩阵上面绘制斜线,角度为45度
def drawLineMatrix(shape=(801,801),angle=45,step = 1 ):
    array =np.zeros(shape)
    arrayCopy = copy.deepcopy(array)
    for i in range(0,array.shape[0],step):
        for j in range(0,array.shape[1],step):
            startRow,startCol = i,j
            while(startRow<shape[0] and startCol<shape[1] and arrayCopy[startRow,startCol]!=255):
                arrayCopy[startRow,startCol]=255
                startRow+=1
                startCol+=1
    return arrayCopy

# 再一个矩阵上面绘制斜点,角度为45度
def drawDotMatrix(shape=(801,801),angle=45,step = 1,dots=12):
    array =np.zeros(shape)
    arrayCopy = copy.deepcopy(array)
    for i in range(0,array.shape[0],step):
        for j in range(0,array.shape[1],step):
            startRow,startCol = i,j
            while(startRow<shape[0] and startCol<shape[1] and arrayCopy[startRow,startCol]!=255):
                arrayCopy[startRow,startCol]=255
                startRow+=dots
                startCol+=dots
    return arrayCopy

# 再一个矩阵上面绘制竖直的线,
def drawLineMatrix_竖直点(shape=(801,801),angle=45,step = 18 ):
    array =np.zeros(shape)
    arrayCopy = copy.deepcopy(array)
    for i in range(0,array.shape[0],step):
        # for j in range(0,array.shape[1],step):
        startRow,startCol = i,0
        while(startRow<shape[0] and startCol<shape[1] and arrayCopy[startRow,startCol]!=255):
            arrayCopy[startRow,startCol]=255
            startCol+=step
    return arrayCopy

# 再一个矩阵上面绘制竖直的线,
def drawLineMatrix_竖直线(shape=(801,801),angle=45,step = 18 ):
    array =np.zeros(shape)
    arrayCopy = copy.deepcopy(array)
    for i in range(0,array.shape[0],step):
        # for j in range(0,array.shape[1],step):
        startRow,startCol = i,0
        while(startRow<shape[0] and startCol<shape[1] and arrayCopy[startRow,startCol]!=255):
            arrayCopy[startRow,startCol]=255
            startCol+=1
    return arrayCopy


def filter加粗轮廓(data,author="wangxutao"):
    a,b = data.shape
    datacopy = copy.deepcopy(data)
    for i in range(1,a-1):
        for j in range(1,b-1):

            if data[i-1,j]==255 or data[i+1,j]==255 or data[i,j-1]==255 or data[i,j+1]==255 :
                datacopy[i,j] = 255
    return datacopy

# 二维数组旋转90度
def matrixRotate(matrixRotate):
    array=matrixRotate
    for i in range(array.shape[0]//2):
        for j in range(array.shape[1]):
            temp = array[i,j]
            array[i,j] =array[ array.shape[0]-i-1,j]
            array[ array.shape[0]-i-1,j]=temp
    return array

def getLine(shape=(801,801),angle=45,step=18):
    array = drawLineMatrix(shape=shape, angle=angle, step=step)
    # save_pic(array, save_place="pic没加粗.png", style="gray")
    array = filter加粗轮廓(array, author="wangxutao")
    array = filter加粗轮廓(array, author="wangxutao")
    # array= filter加粗轮廓(array,author="wangxutao")
    # save_pic(array, save_place="pic加粗.png", style="gray")
    array =matrixRotate(array)
    # save_pic(array, save_place="pic加粗旋转90度.png", style="gray")
    return array

def getDots(shape=(801,801),angle=45,step=18):
    array = drawDotMatrix(shape=shape, angle=angle, step=step)
    # save_pic(array, save_place="pic没加粗.png", style="gray")
    array = filter加粗轮廓(array, author="wangxutao")
    array = filter加粗轮廓(array, author="wangxutao")
    # array= filter加粗轮廓(array,author="wangxutao")
    # save_pic(array, save_place="pic加粗.png", style="gray")
    array =matrixRotate(array)
    # save_pic(array, save_place="pic加粗旋转90度.png", style="gray")
    return array


def 给原始轮廓加上斜线(originCannyArray,maskOrigin,lineArray):
    rows,cols = originCannyArray.shape
    rescopy = copy.deepcopy(originCannyArray)
    
    for i in range(rows):
        for j in range(cols):
            if maskOrigin[i,j]==255:
                continue
            else:
                rescopy[i,j]=lineArray[i,j]+rescopy[i,j]
                if(rescopy[i,j]>255):
                    rescopy[i,j]=255
    return rescopy

def getLine_竖直的点(shape=(801,801),angle=45,step=18):
    array = drawLineMatrix_竖直点(shape=shape, angle=angle, step=step)
    # save_pic(array, save_place="pic没加粗.png", style="gray")
    array = filter加粗轮廓(array, author="wangxutao")
    array = filter加粗轮廓(array, author="wangxutao")
    array= filter加粗轮廓(array,author="wangxutao")
    # save_pic(array, save_place="pic加粗.png", style="gray")
    array =matrixRotate(array)
    # save_pic(array, save_place="pic加粗旋转90度.png", style="gray")
    return array

def getLine_竖直的线(shape=(801,801),angle=45,step=18):
    array = drawLineMatrix_竖直线(shape=shape, angle=angle, step=step)
    # save_pic(array, save_place="pic没加粗.png", style="gray")
    array = filter加粗轮廓(array, author="wangxutao")
    array = filter加粗轮廓(array, author="wangxutao")
    # array= filter加粗轮廓(array,author="wangxutao")
    # save_pic(array, save_place="pic加粗.png", style="gray")
    array =matrixRotate(array)
    # save_pic(array, save_place="pic加粗旋转90度.png", style="gray")
    return array

if __name__ == '__main__':


    data = np.reshape(np.arange(1601**2),(1601,1601))
    returnliSt = splitNumpyAsBlock(data,4)
    print(len(returnliSt))
    lenList = len(returnliSt)
    listheredata = getAllFilesFromDir(r"C:\psoInverseProblem\深度学习的与训练模型生成\savePlace\data_saveDir")
    listherelabel = getAllFilesFromDir(r"C:\psoInverseProblem\深度学习的与训练模型生成\savePlace\label_saveDir")


    data =resizeLable(r"C:\psoInverseProblem范围系数巨大\深度学习的与训练模型生成\savePlace\label_saveDir"+"//0a2e7c96940c484db59b4dcc4ce826f1.pkl_label.pkl")

    print(data)
    x1 = data[0]
    x2 = data[1 ]
    print(x1 , x2)
    print(type(x1))


    # # data =np.array([[1,2,3,4],[,5,6],[7,8,9]])
    # blockSize = 2
    # blockNumber=data.shape[0]//blockSize
    # data = data[:blockSize*blockNumber,:blockSize*blockNumber]
    #
    # data = np.reshape(data,(blockSize,-1))
    # print(data)

