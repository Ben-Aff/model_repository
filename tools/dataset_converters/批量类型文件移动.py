import os
import shutil

def move_files_by_extension(folder_path, folder2_path, extension):
    # 确保文件夹2存在
    if not os.path.exists(folder2_path):
        os.makedirs(folder2_path)

    # 遍历文件夹内所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件的后缀名是否为指定的扩展名
            if file.endswith(extension):
                # 构建源文件路径和目标文件路径
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(folder2_path, file)
                # 移动文件
                shutil.move(src_file_path, dest_file_path)

if __name__ == "__main__":
    # 文件夹路径
    folder_path = "/home/renweilun/桌面/mmdetection/project/Rop3D/newsplit/train"
    # 文件夹2的路径
    folder2_path = "/home/renweilun/桌面/mmdetection/project/Rop3D/newsplit/train/labels"
    # 指定的文件格式后缀
    extension = ".txt"  # 例如，移动所有的 .txt 文件

    # 移动文件夹内指定格式后缀的文件到文件夹2
    move_files_by_extension(folder_path, folder2_path, extension)
