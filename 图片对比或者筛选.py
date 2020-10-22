import matplotlib.pyplot as plt
import os
from skimage import io, transform
# 筛选图片并记录哪些需要保留或者删除
sharp_path = "./sharp_piece"
fusion_path = "./fusion_piece"


img_names = os.listdir(sharp_path)
count  = 0
img_names.sort()
delete_list = []
for name in img_names:
	count =count +1
	sharp_img = io.imread(fusion_path + '/' + name, as_gray = True)
	fusion_img = io.imread(sharp_path + '/' + name, as_gray = True)
	plt.subplot(121), plt.imshow(sharp_img), plt.title(name)
	plt.subplot(122), plt.imshow(fusion_img)
	plt.show()
	plt.close()
	delete = input("Tell me whether delete this picture,input enter to delete")
	print(name,'whether to delete :',delete)
	if delete=='':
		delete_list.append(fusion_path + '/' + name)
	print(delete_list)
print(delete_list)
print('all count = ',count)
# 000703.2700.4740
