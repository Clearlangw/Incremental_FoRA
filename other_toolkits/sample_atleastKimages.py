import os
import shutil
import random
from collections import defaultdict

def select_images_by_instance(base_path, dest_path, categories=['train', 'val', 'test'], K=70, image_extensions=['.jpg', '.png', '.bmp'], base_sensor='rgb'):
    types = ['rgb', 'ir']
    random.seed(42)

    for category in categories:
        # 1. 只基于base_sensor进行抽取
        image_dir = os.path.join(base_path, base_sensor, 'images', category)
        label_dir = os.path.join(base_path, base_sensor, 'labels', category)

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"跳过不存在的目录: {image_dir} 或 {label_dir}")
            continue

        # 统计每张图片每个类别的实例数
        image_to_class_count = defaultdict(lambda: defaultdict(int))
        class_to_images = defaultdict(set)
        all_image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1] in image_extensions]
        for image_file in all_image_files:
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            if not os.path.exists(label_path):
                continue
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = parts[0]
                image_to_class_count[image_file][cls] += 1
                class_to_images[cls].add(image_file)

        # 贪心抽取图片，累计每类实例数≥K
        selected_images = set()
        class_instance_count = defaultdict(int)
        all_images = list(image_to_class_count.keys())
        random.shuffle(all_images)
        #超支的抽取法，因为我只统计当前关心的类够不够，没有维护一个整体的队列
        for cls, images in class_to_images.items():
            needed = K - class_instance_count[cls]
            if needed <= 0:
                continue
            candidate_images = list(images - selected_images)
            candidate_images.sort(key=lambda img: image_to_class_count[img][cls], reverse=True)
            for img in candidate_images:
                if class_instance_count[cls] >= K:
                    break
                selected_images.add(img)
                class_instance_count[cls] += image_to_class_count[img][cls]
            if class_instance_count[cls] < K:
                print(f"{category}-{base_sensor}类别{cls}实例数不足{K}，仅有{class_instance_count[cls]}个实例！")

        print(f"{category}-{base_sensor}最终选中图片数：{len(selected_images)}")

        # 统计最终选中图片中每个类别的实例数
        final_class_count = defaultdict(int)
        for image_file in selected_images:
            for cls, count in image_to_class_count[image_file].items():
                final_class_count[cls] += count

        print(f"{category}集最终各类别实例数：")
        for cls, count in final_class_count.items():
            print(f"类别{cls}: {count}")

        # 2. 对两个sensor_type都只复制selected_images中的图片和标签
        for sensor_type in types:
            image_dir = os.path.join(base_path, sensor_type, 'images', category)
            label_dir = os.path.join(base_path, sensor_type, 'labels', category)
            image_dest_dir = os.path.join(dest_path, sensor_type, 'images', category)
            label_dest_dir = os.path.join(dest_path, sensor_type, 'labels', category)
            os.makedirs(image_dest_dir, exist_ok=True)
            os.makedirs(label_dest_dir, exist_ok=True)

            for image_file in selected_images:
                src_image = os.path.join(image_dir, image_file)
                src_label = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
                dst_image = os.path.join(image_dest_dir, image_file)
                dst_label = os.path.join(label_dest_dir, os.path.splitext(image_file)[0] + '.txt')
                if os.path.exists(src_image):
                    shutil.copy(src_image, dst_image)
                else:
                    print(f"警告: {sensor_type}缺失图片 {src_image}")
                if os.path.exists(src_label):
                    shutil.copy(src_label, dst_label)
                else:
                    print(f"警告: {sensor_type}缺失标签 {src_label}")

# 用法
base_path = '/root/autodl-tmp/dv_vedai'
dest_path = base_path + '_fewshot_classmin250'
categories = ['train']
#categories = ['train', 'val', 'test']
select_images_by_instance(base_path, dest_path, categories=categories, K=250)