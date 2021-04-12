# encoding=utf-8
import numpy as np

if __name__ == '__main__':
    import h5py
    m1= ("U://大创for学妹//可以使用的数据4-13//"+"matlabtest"+str(70)+".mat")
    m70 = ("U://大创for学妹//可以使用的数据4-13//"+"matlabtest"+str(1)+".mat")

    

    mat1 = h5py.File(m1)
    mat70 = h5py.File(m70)
    # 各个key查看
    print( mat.keys())
    # 各个value查看
    print(mat.values())
    pic_data= mat['Efarfield']
    # 转换道npy文件
    picdata70=np.transpose(pic_data)
    print(mat['Efarfield'].shape)

    
