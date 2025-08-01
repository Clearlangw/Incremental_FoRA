import os
import shutil
import random


def copy_random_files(base_path, num_files=500):
    # categories = ['train', 'val', 'test']
    categories = ['test']
    types = ['rgb', 'ir']
    components = ['images', 'labels']
    image_extensions = ['.jpg', '.png', '.bmp']  # 支持的图像文件类型

    # 为随机选择设置种子，确保可重复性
    random.seed(42)

    # 在 base_path 后添加 _random100 后缀
    # new_base_path = f"{base_path}_fullvedai"
    new_base_path = os.path.join(os.path.dirname(base_path), "dv_vedai_small_test")
    # 遍历每种类别（train, val, test）
    for category in categories:
        # 生成两种文件类型（images 和 labels）及两种传感器类型（rgb 和 ir）的路径
        paths = {
            (sensor_type, component): os.path.join(base_path, sensor_type, component, category)
            for sensor_type in types
            for component in components
        }

        # 确保所有路径都存在
        if not all(os.path.exists(path) for path in paths.values()):
            print(f"Error: Some paths in {paths.values()} do not exist.")
            continue

        # 获取所有可能的文件名
        # all_image_files = [f for f in os.listdir(paths[('rgb', 'images')]) if
        #                    os.path.splitext(f)[1] in image_extensions]

        # # 随机选择 num_files 个文件
        # if len(all_image_files) < num_files:
        #     print(f"Warning: Not enough files in {paths[('rgb', 'images')]} to select {num_files}.")
        #     continue
        # selected_files = random.sample(all_image_files, num_files)
        all_image_files = [f for f in os.listdir(paths[('rgb', 'images')]) if
                           os.path.splitext(f)[1] in image_extensions]

        # 先分为含vedai和不含vedai的两组
        vedai_files = [f for f in all_image_files if 'vedai' in f]
        non_vedai_files = [f for f in all_image_files if 'vedai' not in f]

        # 计算各自要抽取的数量
        # num_vedai = int(num_files * 1) #原来0.6
        # num_non_vedai = num_files - num_vedai
        #num_vedai = len(vedai_files)
        num_vedai = int(num_files * 0.8)
        num_non_vedai = num_files - num_vedai

        if len(vedai_files) < num_vedai or len(non_vedai_files) < num_non_vedai:
            print(f"Warning: Not enough vedai or non-vedai files in {paths[('rgb', 'images')]} to select {num_files}.")
            continue

        selected_vedai = random.sample(vedai_files, num_vedai)
        selected_non_vedai = random.sample(non_vedai_files, num_non_vedai)
        selected_files = selected_vedai + selected_non_vedai
        label_files = [f"{os.path.splitext(f)[0]}.txt" for f in selected_files]

        # 对每个 sensor_type 进行文件复制
        for sensor_type in types:
            # 定义源目录和目标目录
            image_source_dir = paths[(sensor_type, 'images')]
            label_source_dir = paths[(sensor_type, 'labels')]

            image_dest_dir = os.path.join(new_base_path, sensor_type, 'images', category)
            label_dest_dir = os.path.join(new_base_path, sensor_type, 'labels', category)

            # 创建目标文件夹
            os.makedirs(image_dest_dir, exist_ok=True)
            os.makedirs(label_dest_dir, exist_ok=True)

            # 复制 images 文件
            for file in selected_files:
                shutil.copy(os.path.join(image_source_dir, file), os.path.join(image_dest_dir, file))

            # 复制 labels 文件
            for file in label_files:
                shutil.copy(os.path.join(label_source_dir, file), os.path.join(label_dest_dir, file))


# 调用函数
base_path = '/root/autodl-tmp/dv_vedai'  # 修改为你的基本路径
copy_random_files(base_path)
