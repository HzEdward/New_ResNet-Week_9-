'''
本py文件用于生成一个列表，其中包含所有的在Final Dataset中的图片的路径
'''
import os
import sys

def filter(csv_path="./data.csv", dataset_path="../Final Dataset/"):
    for data_split in ["train", "val"]:
        for label in ["blacklist", "whitelist"]:
            label_path = os.path.join(dataset_path, data_split, label)
            for folder in os.listdir(label_path):
                if folder.startswith('.'):
                    continue
                else:
                    folder_path = os.path.join(label_path, folder)

                    for file in os.listdir(folder_path):
                        if file.startswith('image') and file.endswith('.png'):
                            rgb_image_path = os.path.join(folder_path, file)
                        