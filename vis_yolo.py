import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os


def visualize_and_save_labels(image_path, label_path, output_path):
    # 打开图片文件
    image = Image.open(image_path)
    width, height = image.size

    # 准备绘图
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 读取标签文件
    with open(label_path, 'r') as file:
        labels = file.readlines()
    print(len(labels))
    for label in labels:
        parts = label.split()
        category = parts[0]
        x_center, y_center, w, h = map(float, parts[1:5])
        if float(parts[-1]) >= 0.001:
            # 将x, y, w, h转换为绝对坐标系统
            x = (x_center - w / 2) * width
            y = (y_center - h / 2) * height
            w *= width
            h *= height

            # 创建一个矩形patch
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            # 添加patch到Axes中
            ax.add_patch(rect)
            plt.text(x, y, category, color='white', fontsize=12, verticalalignment='top', bbox={'color': 'red', 'pad': 0})

    plt.axis('off')  # 关闭坐标轴
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # 保存图片到文件
    # plt.show()


def visualize_folder(images_folder, labels_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历images文件夹中的所有图片文件
    for image_file in os.listdir(images_folder):
        # 构建图片和标签文件的完整路径
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, f"{os.path.splitext(image_file)[0]}.txt")
        output_path = os.path.join(output_folder, image_file)
        
        # 检查是否存在对应的标签文件
        if os.path.exists(label_path):
            # 调用visualize_and_save_labels函数生成标注图片
            visualize_and_save_labels(image_path, label_path, output_path)
        else:
            print(f"Warning: No label file found for {image_file}")

# 调用函数
visualize_folder("/root/autodl-tmp/fewshot_incremental_dv_vedai/rgb/images/train/", "/root/autodl-tmp/fewshot_incremental_dv_vedai/rgb/labels/train/", "/root/autodl-tmp/fewshot_incremental_dv_vedai/rgb/vis_check")