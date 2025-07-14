# global_mode.py
# global_mode.py
import os

global_mode = 'rgb'  # 初始化全局变量
file_path = 'global_mode.txt'  # 写入路径


def set_global_mode(value):
    global global_mode
    global_mode = value
    # 将值写入指定路径
    with open(file_path, 'w') as f:
        f.write(global_mode)


def read_global_mode():
    # 从文件中读取全局变量的值
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read().strip()  # 返回去掉空白的值
    return global_mode  # 如果文件不存在，返回默认值
