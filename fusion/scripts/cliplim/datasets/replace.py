import json
import os

# 获取当前目录下的文件路径
file_path = os.path.join(os.getcwd(), 'en_val.json')

# 读取json文件
with open(file_path, 'r') as file:
    data = json.load(file)

# 将数据中的所有\\替换为/
updated_data = json.dumps(data).replace('\\\\', '/')

# 写入更新后的数据到同一个文件
with open(file_path, 'w') as file:
    file.write(updated_data)

print("替换完成。")