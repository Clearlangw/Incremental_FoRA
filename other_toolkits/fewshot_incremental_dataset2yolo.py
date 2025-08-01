import os
import shutil
from PIL import Image

# 基础配置
source_base_dir = '/root/autodl-tmp/shot85/'  # 原始数据集位置(即原来的shot85)
target_base_dir = '/root/autodl-tmp/wenhao_fewshot_incremental_dv_vedai/'  # 新数据集位置
nameidmap = {'car': 0, 'truck': 1, 'bus': 2, 'feright_car': 3, 'van': 4,'pick-up':5,
    'camping-car':6,'tractor':7}  # 类名到ID的映射

# 创建新的目录结构
os.makedirs(target_base_dir, exist_ok=True)
for part in ['train']:
    for mode in ['rgb', 'ir']:
        os.makedirs(os.path.join(target_base_dir, mode, 'images', part), exist_ok=True)
        os.makedirs(os.path.join(target_base_dir, mode, 'labels', part), exist_ok=True)


# 文件复制和标签转换函数
def convert_and_copy_files(part, mode):
    source_image_dir = os.path.join(source_base_dir, mode,'images') #这里没有part了
    source_label_dir = os.path.join(source_base_dir, mode,'labelTxt')
    target_image_dir = os.path.join(target_base_dir, mode, 'images', part)
    target_label_dir = os.path.join(target_base_dir, mode, 'labels', part)

    for filename in os.listdir(source_image_dir):
        # 复制图像文件
        src_img_path = os.path.join(source_image_dir, filename)
        tgt_img_path = os.path.join(target_image_dir, filename)
        shutil.copy(src_img_path, tgt_img_path)

        # 处理对应的标签文件
        base_filename = filename.split('.')[0] + '.txt'
        src_label_path = os.path.join(source_label_dir, base_filename)
        tgt_label_path = os.path.join(target_label_dir, base_filename)

        if os.path.exists(src_label_path):
            # 读取图像尺寸
            with Image.open(src_img_path) as img:
                iw, ih = img.size

            # 转换标签文件
            with open(src_label_path, 'r') as file:
                lines = file.readlines()

            new_labels = []
            for line in lines:
                parts = line.strip().split()
                coords = list(map(float, parts[:8]))
                class_name, difficulty = parts[8], int(parts[9])
                class_id = nameidmap[class_name]

                # 计算bounding box的中心点和尺寸
                xmin = min(coords[0::2])
                xmax = max(coords[0::2])
                ymin = min(coords[1::2])
                ymax = max(coords[1::2])
                w = xmax - xmin
                h = ymax - ymin
                cx = xmin + w / 2
                cy = ymin + h / 2

                # 转换为相对坐标
                cx /= iw
                cy /= ih
                w /= iw
                h /= ih

                new_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            # 写入新的标签文件
            with open(tgt_label_path, 'w') as file:
                file.writelines(new_labels)


# 处理train, val, test目录
for part in ['train']:
    for mode in ['rgb', 'ir']:
        convert_and_copy_files(part, mode)

print("数据转换完成！")
