# encoding=utf-8
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils

if __name__ == '__main__':

    d1datanpy = "1dgetcanny轮廓.npy"
    data = np.load(d1datanpy)
    # img = cv2.imread(r"eall_ey.npy400_region.png",0)

    img = data
    img = img.astype(np.uint8)

    # cv2.Canny(data , 0,255)
    # v1  =cv2.Canny(img , 100,200)
    edges = cv2.Canny(img, 100, 200)


    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    utils.savenpyasexcel(edges,d1datanpy.replace(".npy","border.xlsx"))
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
