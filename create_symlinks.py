#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import glob

def create_symlinks(source_dir, target_dir):
    """
    将源目录下的文件通过软链接形式补充到目标目录中
    
    Args:
        source_dir: 源目录路径 (文件夹A)
        target_dir: 目标目录路径 (文件夹B)
    """
    # 确保路径存在
    if not os.path.exists(source_dir):
        print(f"错误：源目录 {source_dir} 不存在")
        return
    
    if not os.path.exists(target_dir):
        print(f"错误：目标目录 {target_dir} 不存在")
        return
    
    # 创建目标目录下的子目录结构（如果不存在）
    for data_type in ['images', 'labels']:
        for split in ['train', 'test']:
            target_subdir = os.path.join(target_dir, data_type, split)
            if not os.path.exists(target_subdir):
                os.makedirs(target_subdir)
                print(f"创建目录: {target_subdir}")
    
    # 处理images目录
    for split in ['train', 'test']:
        source_images_dir = os.path.join(source_dir, 'images', split)
        target_images_dir = os.path.join(target_dir, 'images', split)
        
        if not os.path.exists(source_images_dir):
            print(f"警告：源目录 {source_images_dir} 不存在，跳过")
            continue
        
        # 获取源目录中的所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(source_images_dir, ext)))
        
        # 创建软链接
        for image_file in image_files:
            image_name = os.path.basename(image_file)
            target_link = os.path.join(target_images_dir, image_name)
            
            # 如果目标链接已存在，则跳过
            if os.path.exists(target_link):
                print(f"跳过: {target_link} 已存在")
                continue
            
            # 创建软链接（使用绝对路径）
            os.symlink(image_file, target_link)
            print(f"创建软链接: {image_file} -> {target_link}")
    
    # 处理labels目录
    for split in ['train', 'test']:
        source_labels_dir = os.path.join(source_dir, 'labels', split)
        target_labels_dir = os.path.join(target_dir, 'labels', split)
        
        if not os.path.exists(source_labels_dir):
            print(f"警告：源目录 {source_labels_dir} 不存在，跳过")
            continue
        
        # 获取源目录中的所有标签文件
        label_files = glob.glob(os.path.join(source_labels_dir, '*.txt'))
        
        # 创建软链接
        for label_file in label_files:
            label_name = os.path.basename(label_file)
            target_link = os.path.join(target_labels_dir, label_name)
            
            # 如果目标链接已存在，则跳过
            if os.path.exists(target_link):
                print(f"跳过: {target_link} 已存在")
                continue
            
            # 创建软链接（使用绝对路径）
            os.symlink(label_file, target_link)
            print(f"创建软链接: {label_file} -> {target_link}")

if __name__ == "__main__":
    # 设置源目录和目标目录
    source_dir = "/root/autodl-tmp/IR_DroneVehicle_v8"
    target_dir = "/root/autodl-tmp/IR_car_and_plane"
    
    print(f"开始将 {source_dir} 的文件软链接到 {target_dir}")
    create_symlinks(source_dir, target_dir)
    print("处理完成") 