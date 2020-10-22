# 遍历得到某个文件夹下的所有文件名
def getFileName(dirName):
    import os
    fileList= []
    filePath = dirName
    for i, j, k in os.walk(filePath):
        # i是当前路径，j得到文件夹名字，k得到文件名字
        print(i, j, k)
        fileList.append(k)
    return fileList[0]
# 特定的文件后缀保存
def data_select(data_dir):  #  
    import  glob
    file_list = list(glob.glob(data_dir + '/*.png')) + list(glob.glob(data_dir + '/*.jpg'))   # get name list of all .png files
    data = []
    print(file_list) # 得到文件的路径列表
    return file_list

if __name__ == '__main__':
    dir = '../gray_origin'
    flist_to_writed = 'test.flist'
    sum_file =str('')
    pic_list = getFileName(dir)
    for i in pic_list:
        temp=''
        temp=temp+dir+'/'+str(i)+'\n'
        sum_file = sum_file+(temp)
    f1 = open(flist_to_writed, 'w')
    f1.write(sum_file)
    f1.close()
