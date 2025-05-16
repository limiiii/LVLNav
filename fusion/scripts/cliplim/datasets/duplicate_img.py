#!/home/yang/anaconda3/envs/pytorch3.8/bin/python
# -*- coding: utf-8 -*-
import shutil
import os

# 原始图片路径
image_path = "target_image_depth3.jpg"

# 目标文件名前缀
target_prefix = "small_house_target_image333"

# 复制图片
for i in range(1, 41):
    target_filename = f"{target_prefix}{i}.jpg"
    shutil.copy(image_path, target_filename)
    print(f"复制图片 {image_path} 到 {target_filename} 完成")

# 打印完成提示
print("复制完成")
