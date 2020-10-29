# 特定的文件后缀保存，筛选后的特定后缀
def data_select(data_dir,*args):  #
    import  glob
    # *args是要筛选的的后缀名称可以选择多个
    length = len(args)
    file_list = []
    for i in range(length):
        file_list = file_list + list(glob.glob(data_dir + '/*.'+str(args[i])))
    # file_list = list(glob.glob(data_dir + '/*.png')) + list(glob.glob(data_dir + '/*.jpg'))   # get name list of all .png files
    # data = []
    # print(file_list) # 得到文件的路径列表
    return file_list

import matplotlib.pyplot as plt
import os
from skimage import io, transform
# 筛选图片并记录哪些需要保留或者删除
sharp_path = "./sharp_piece"
fusion_path = "./fusion_piece"

if __name__ == '__main__':
    img_list =data_select('./images','png')
    lenght = len(img_list)//3
    count =0
    for i in range(lenght):
        count = count+1
        sharp_img = io.imread(img_list[i*3+1] , as_gray = True)
        fusion_img = io.imread(img_list[i*3+0], as_gray = True)
        last_img = io.imread(img_list[i*3+1+1], as_gray = True)
        plt.subplot(131), plt.imshow(sharp_img), plt.title('to inpainted')
        plt.subplot(132), plt.imshow(fusion_img),plt.title('after inpainted')
        plt.subplot(133), plt.imshow(last_img),plt.title('real picture')
        plt.show()
        plt.close()


