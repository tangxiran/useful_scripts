def data_select(data_dir):  #
    import  glob
    file_list = list(glob.glob(data_dir + '/*.png')) + list(glob.glob(data_dir + '/*.jpg'))   # get name list of all .png files
    data = []
    print(file_list) # 得到文件的路径列表
    return file_list

if __name__ == '__main__':
    # 扩大轮廓的外边缘几个像素
    filelist  =data_select('../masks')
    num_point = 1

    for file in filelist:

        # 扩大轮廓的外边缘几个像素
        file_mask  = file
        for i in range(num_point):
            import cv2

            mask = cv2.imread(filename=file_mask, flags=0)
            print(mask)
            height, width = mask.shape
            mask_copy = mask.copy()
            for i in range(height - 1):
                for j in range(width - 1):
                    if mask[i, j] == 0:
                        if (mask[i - 1, j - 1] == 255 or mask[i - 1, j - 0] == 255 or mask[i - 1, j + 1] == 255 or mask[
                            i - 0, j - 1] == 255 or mask[i - 0, j + 1] == 255 or mask[i + 1, j - 1] == 255 or mask[
                            i + 1, j + 0] == 255 or mask[i + 1, j + 1] == 255):
                            mask_copy[i, j] = 255
            mask_copy = mask_copy
            print(mask_copy)
            cv2.imshow('realmask', mask)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.imshow('biggermask', mask_copy)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            save_place= file.replace('masks','masks//thin')
            cv2.imwrite(filename=save_place,img=mask_copy)