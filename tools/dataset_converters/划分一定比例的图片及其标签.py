import os
import random
import shutil

def move_images_with_labels(folder1_path, folder2_path, proportion):
    # 获取文件夹1中的所有文件和文件夹
    files = os.listdir(folder1_path)
    # 从文件列表中筛选出图片和对应的标签
    image_files = [file for file in files if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')]
    label_files = [file for file in files if file.endswith('.txt')]

    # 将文件按照文件名前缀排序，以确保图片和标签的一致性
    image_files.sort()
    label_files.sort()

    # 确保文件夹2存在
    if not os.path.exists(folder2_path):
        os.makedirs(folder2_path)

    # 计算要移动的文件数量
    num_files_to_move = int(len(image_files) * proportion)

    # 随机选择一定比例的文件
    files_to_move = random.sample(range(len(image_files)), num_files_to_move)

    # 移动选定的文件和对应的标签
    for idx in files_to_move:
        # 源文件路径
        src_image_path = os.path.join(folder1_path, image_files[idx])
        src_label_path = os.path.join(folder1_path, label_files[idx])
        # 目标文件路径
        dest_image_path = os.path.join(folder2_path, image_files[idx])
        dest_label_path = os.path.join(folder2_path, label_files[idx])
        # 移动文件
        shutil.move(src_image_path, dest_image_path)
        shutil.move(src_label_path, dest_label_path)

if __name__ == "__main__":
    # 文件夹1的路径
    folder1_path = "/home/renweilun/桌面/mmdetection/project/Rop3D/oldsplit/train2017"
    # 文件夹2的路径
    folder2_path = "/home/renweilun/桌面/mmdetection/project/Rop3D/newsplit/train2017"
    # 选择的比例
    proportion = 0.1  # 例如，选择一半的图片

    # 移动图片和标签
    move_images_with_labels(folder1_path, folder2_path, proportion)
