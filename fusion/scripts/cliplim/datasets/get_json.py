#!/home/yang/anaconda3/envs/pytorch3.8/bin/python
# -*- coding: utf-8 -*-
import os
import json
# 当前路径
current_path = os.getcwd()
files = os.listdir(current_path)
image_files = [file for file in files if file.endswith((".jpg"))]
ans = []
for file in image_files:
    item = {}
    imagee_file_name = "./cam1/data/" + file
    depthimage_file_name = "./depth1/data/" + file
    item["image"] = imagee_file_name
    item["caption"] = ["sofa"]
    item["depth_image"] = depthimage_file_name
    ans.append(item)
print(ans)
with open("train.json", "w") as f:
    json.dump(ans, f)

print("保存完成")
