import os
"""
这个脚本将删除文件夹一中与文件夹二中相同文件名的文件。

"""
# 定义两个文件夹的路径
folder1_path = "/home/renweilun/project/mmdetection/data/newbaseline/test/test2_54_80"
folder2_path = "/home/renweilun/project/mmdetection/data/newbaseline/test/test3_21_31"

# 获取文件夹1和文件夹2中的所有文件名
folder1_files = os.listdir(folder1_path)
folder2_files = os.listdir(folder2_path)

# 转换文件名为集合以便进行比较
folder1_files_set = set(folder1_files)
folder2_files_set = set(folder2_files)

# 找到两个文件夹中相同的文件名
common_files = folder1_files_set.intersection(folder2_files_set)

# # 删除文件夹1中与文件夹2中相同文件名的文件
# for file_name in common_files:
#     file1_path = os.path.join(folder1_path, file_name)
#     if os.path.exists(file1_path) and os.path.isfile(file1_path):
#         os.remove(file1_path)
#         print("已删除文件:", file1_path)

