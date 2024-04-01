"""
这个脚本会遍历文件夹2中的JSON和JPG文件，检查它们的文件名前缀是否与文件夹1中的JPG文件匹配。
如果匹配，它将移动JSON文件到文件夹1中，同时也将同名的JPG文件移动到文件夹1中


"""
import os
import shutil

# 定义两个文件夹的路径
folder1_path = "/home/renweilun/project/mmdetection/data/baseline/json1_general290"
folder2_path = "/home/renweilun/project/mmdetection/data/GT_2/val"

# 获取文件夹1中的所有JPG文件名（不包含扩展名）
jpg_files = {os.path.splitext(f)[0] for f in os.listdir(folder1_path) if f.endswith(".jpg")}

# 遍历文件夹2，查找匹配的JSON和JPG文件，并移动到文件夹1中
for file_name in os.listdir(folder2_path):
    # 获取文件名的前缀部分（不包含扩展名）
    file_prefix, file_extension = os.path.splitext(file_name)
    if file_extension == ".json" and file_prefix in jpg_files:
        # JSON文件与文件夹1中的JPG文件同名前缀
        json_file_path = os.path.join(folder2_path, file_name)
        jpg_file_path = os.path.join(folder1_path, f"{file_prefix}.json")

        # 移动JSON文件到文件夹1中
        shutil.move(json_file_path, jpg_file_path)
        print(f"已移动文件: {file_name} 到 {folder1_path}")

    elif file_extension == ".jpg" and file_prefix in jpg_files:
        # JPG文件与文件夹1中的JPG文件同名前缀
        jpg_file_path = os.path.join(folder2_path, file_name)
        new_jpg_file_path = os.path.join(folder1_path, file_name)

        # 移动JPG文件到文件夹1中
        shutil.move(jpg_file_path, new_jpg_file_path)
        print(f"已移动文件: {file_name} 到 {folder1_path}")
