import imageio


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return 0
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
    file_list =sorted(file_list,key = lambda i:len(i),reverse=False)
    return file_list

    # return file_list.sort(key = lambda i:len(i),reverse=True)

def main():
    image_list = data_select('./','png')
    gif_name = 'cat.gif'
    duration = 0.02
    create_gif(image_list, gif_name, duration)


if __name__ == '__main__':
    main()
