import os
'''
这个代码会在给定的文件夹中查找所有以 .jpg 后缀结尾的文件，提取它们的文件名前缀，
然后将这些前缀写入一个新建的 txt 文件中。你需要将 "path/to/folder" 替换为实际的文件夹路径，并指定你想要创建的 txt 文件的路径 "path/to/output.txt"。
'''


def write_image_prefixes_to_txt(folder_path, txt_file_path):
    # 获取文件夹中所有以 .jpg 后缀结尾的文件
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]

    # 提取文件名前缀
    image_prefixes = [file.split('.')[0] for file in image_files]

    # 将前缀写入 txt 文件
    with open(txt_file_path, 'w') as txt_file:
        for prefix in image_prefixes:
            txt_file.write(prefix + '\n')

if __name__ == "__main__":
    # 文件夹路径
    folder_path = "/home/renweilun/桌面/mmdetection/project/Rop3D/newsplit/train2017"
    # 新建的 txt 文件路径
    txt_file_path = "/home/renweilun/桌面/mmdetection/project/Rop3D/newsplit/train.txt"

    # 写入图片前缀到 txt 文件
    write_image_prefixes_to_txt(folder_path, txt_file_path)
