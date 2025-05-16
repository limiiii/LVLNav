import json
import os
import ast

categories_list = ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture",
              "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor", "clothes",
              "ceiling", "books", "refridgerator", "television", "paper", "towel", "shower", "box", "whiteboard",
              "person", "night", "toilet", "sink", "lamp", "bathtub", "bag", "otherstructure", "otherfurniture",
              "otherprop"]

def take_annotaion_from_scene(labelpath):
    with open(labelpath, 'r') as f:
        data = json.load(f)
    images_list = data['images']
    annotations_list = data['annotations']
    result_list = []

    img_dict = None
    pri_img_id = None

    for anno in annotations_list:
        if pri_img_id is None or pri_img_id != anno['image_id']:
            if img_dict is not None:
                result_list.append(img_dict)
            img_dict = {'img_name': '', 'label_id': []}
            pri_img_id = anno['image_id']

            for img in images_list:
                if img['id'] == anno['image_id']:
                    img_dict['img_name'] = img['file_name']
        if anno['label_id'] not in img_dict['label_id']:
            img_dict['label_id'].append(anno['label_id'])

    if img_dict is not None:
        result_list.append(img_dict)

    return result_list

#刪除沒有標記信息的圖片，減少數據量
def delete_img(result_list, img_dir):
    img_files = os.listdir(img_dir)
    print(img_files)
    img_names = [result['img_name'] for result in result_list]
    count = 0
    total = 0
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        total += 1
        if img_path not in img_names:
            os.remove(img_path)
            count += 1
            print(f"Deleted {img_path}")

    print(f"Deletion completed! {count} image deleted!,total files:{total}")
#刪除圖片后，修改train.txt中存儲的標簽内容
def clean_train_file(file_path):
    with open(file_path, 'r') as file:
        train_text = file.read()
        train_text = train_text[1:-1]

    # 將每個字典轉換為 Python 字典對象
    train_text = train_text.replace(' ', '').replace('\n', '')
    train_text = train_text.replace('img_name', 'image').replace('label_id', 'caption')
    # 切割字符串，每個字典作為一個元素
    dicts_list = train_text.split('}{')

    # 刪除多餘的字典內容
    result_list = []
    for d in dicts_list:
        #转换成字典对象
        d = ast.literal_eval('{' + d + '}')
        #将字典中caption的数字标签改为英文
        caption_list = d['caption']
        d['caption'] = [categories_list[num-1] for num in caption_list]
        #删除已被删除的图片的标签，并添加深度图的字典信息
        img_name = d['image']
        if img_name is not None and os.path.exists(img_name):
            d['depth_image'] = img_name.replace("cam0", "depth0")
            result_list.append(d)
    # 返回剩餘字典的數量
    print("Remaining dictionaries: ", len(result_list))

    with open('train.json', 'w') as file:
        json.dump(result_list, file)
    print("Writing to file done!!")
    os.remove(file_path)

def keep_one_delete_two(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Invalid directory path: {directory_path}")
        return

    image_files = sorted([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])

    for i in range(0, len(image_files), 4):
        if i + 3 < len(image_files):
            keep_file = image_files[i]
            delete_file_1 = image_files[i + 1]
            delete_file_2 = image_files[i + 2]
            delete_file_3 = image_files[i + 3]

            keep_file_path = os.path.join(directory_path, keep_file)
            delete_file_path_1 = os.path.join(directory_path, delete_file_1)
            delete_file_path_2 = os.path.join(directory_path, delete_file_2)
            delete_file_path_3 = os.path.join(directory_path, delete_file_3)

            print(f"Keeping: {keep_file_path}")
            print(f"Deleting: {delete_file_path_1}")
            print(f"Deleting: {delete_file_path_2}")

            # 執行刪除操作
            os.remove(delete_file_path_1)
            os.remove(delete_file_path_2)
            os.remove(delete_file_path_3)

    print("Operation completed successfully.")
def change_path2val(path):
    file_path = path

    # 读取 train.json 文件内容
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    # 替换字符串
    json_data = [
        {
            key: value.replace("cam0", "valcam0").replace("depth0", "valdepth0")
            if isinstance(value, str)
            else value
            for key, value in item.items()
        }
        for item in json_data
    ]

    # 将修改后的内容写回文件
    with open('val.json', 'w') as file:
        json.dump(json_data, file)
    os.remove('train.json')
    os.rename('cam0', 'valcam0')
    os.rename('depth0', 'valdepth0')
    print("String replacements completed.")

def add(new_filename):
    rename_list = []
    with open('train.json', 'r') as file:
        data = json.load(file)
        for item in data:
            dict = {}
            directory = os.path.dirname(item['image'])
            filename = os.path.basename(item['image'])
            img_new_filename = new_filename + filename
            new_image_path = os.path.join(directory, img_new_filename)
            new_image_path = new_image_path.replace('\\', '/')
            os.rename(item['image'], new_image_path)
            dict['image'] = new_image_path
            dict['caption'] = item['caption']

            depdirectory = os.path.dirname(item['depth_image'])
            dep_img_new_filename = new_filename + filename
            depnew_image_path = os.path.join(depdirectory, dep_img_new_filename)
            depnew_image_path = depnew_image_path.replace('\\', '/')
            os.rename(item['depth_image'], depnew_image_path)
            dict['depth_image'] = depnew_image_path
            rename_list.append(dict)
    with open('train.json', 'w') as f:
        json.dump(rename_list, f)
    print('write done!!')

if __name__ == '__main__':
    # #需要讀取的數據標注文件
    # label_path = 'cocolabel.json'
    # img_dir = './cam0/data/'
    # result_list = take_annotaion_from_scene(label_path)
    # #將提取的圖像路徑及其標注信息記錄在train.txt文件中
    # with open('train.txt', 'a') as file:
    #     for item in result_list:
    #         file.write(str(item))
    # delete_img(result_list, img_dir)
    #
    # img_dir = './cam0/data/'
    # keep_one_delete_two(img_dir)
    # depth_img_dir = './depth0/data/'
    # keep_one_delete_two(depth_img_dir)


    # 3.修改.txt内容防止報錯，再執行同步標簽操作
    txt_path = 'train.txt'
    clean_train_file(txt_path)
    # 4.将训练路径修改为验证路径
    change_path2val('train.json')
    # 5 图片名称添加所属数据集
    new_filename = '3FO4JXKO1V6Q04'
    add(new_filename)
