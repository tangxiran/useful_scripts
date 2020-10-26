import imageio


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return
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

def main():
    image_list = ['cat/1.png', 'cat/2.png', 'cat/3.png', 'cat/4.png', 'cat/5.png', 'cat/6.png']
    gif_name = 'cat/cat.gif'
    duration = 0.35
    create_gif(image_list, gif_name, duration)


if __name__ == '__main__':
    main()
