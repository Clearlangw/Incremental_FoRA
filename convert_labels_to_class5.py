#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import glob

def convert_labels_to_class5(labels_dir):
    """
    将指定文件夹下所有YOLO格式标注文件的类别（每行第0个元素）改为5
    
    Args:
        labels_dir: 标注文件所在的目录路径
    """
    # 确保路径存在
    if not os.path.exists(labels_dir):
        print(f"错误：目录 {labels_dir} 不存在")
        return
    
    # 获取所有txt文件
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if not label_files:
        print(f"警告：在 {labels_dir} 中没有找到txt文件")
        return
    
    # 处理每个标注文件
    total_files = len(label_files)
    modified_files = 0
    modified_lines = 0
    
    for file_path in label_files:
        file_modified = False
        new_lines = []
        
        # 读取文件内容
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                new_lines.append(line)
                continue
                
            parts = line.split()
            if len(parts) >= 5:  # YOLO格式应该至少有5个元素：类别和四个坐标值
                # 将第一个元素（类别）改为5
                # print(type(parts[0]))
                # print(parts[0])
                # import sys 
                # sys.exit()
                parts[0] = "5"
                new_line = " ".join(parts)
                new_lines.append(new_line)
                modified_lines += 1
                file_modified = True
            else:
                # 保持原始行不变
                new_lines.append(line)
        
        # 只有当文件被修改时才写入
        if file_modified:
            with open(file_path, 'w') as f:
                f.write('\n'.join(new_lines))
            modified_files += 1
    
    print(f"处理完成：共处理 {total_files} 个文件，修改了 {modified_files} 个文件中的 {modified_lines} 行标注")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将YOLO格式标注文件的类别改为5")
    parser.add_argument("labels_dir", help="标注文件所在的目录路径")
    args = parser.parse_args()
    
    convert_labels_to_class5(args.labels_dir) 