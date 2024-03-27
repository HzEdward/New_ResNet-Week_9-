import os
import pandas as pd

'''
本py文件用于生成一个新的csv文件，其中包含所有的在Final Dataset中的图片的路径
'''

def filter(csv_path="./data.csv", dataset_path="../Final Dataset/"):
    rgb_image_paths = []
    
    for data_split in ["train"]:
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
                            
                            # ../Final Dataset/train/blacklist/296_blacklist_pair/image_Video24_frame007360.png 变成 image_Video24_frame007360
                            rgb_image_path = rgb_image_path.split('/')[-1].split('.')[0] + '.png'
                            
                            # 如果遇到image_Video1_frame000110_shifted.png，就变成image_Video1_frame000110.png, 同理，rotated, flipped, replaced.
                            if rgb_image_path.endswith('_shifted.png'):
                                rgb_image_path = rgb_image_path.split('_shifted')[0] + '.png'
                            elif rgb_image_path.endswith('_rotated.png'):
                                rgb_image_path = rgb_image_path.split('_rotated')[0] + '.png'
                            elif rgb_image_path.endswith('_flipped.png'):
                                rgb_image_path = rgb_image_path.split('_flipped')[0] + '.png'
                            elif rgb_image_path.endswith('_replaced.png'):
                                rgb_image_path = rgb_image_path.split('_replaced')[0] + '.png'
                                
                            if rgb_image_path not in rgb_image_paths:
                                rgb_image_paths.append(rgb_image_path)
    
    return rgb_image_paths
import pandas as pd
import sys

def modify_csv(csv_path="./data copy.csv", rgb_image_paths=[]):
    '''
    本函数用于修改CSV文件，删除不在rgb_image_paths列表中的行
    参数:
        csv_path: CSV文件路径，默认为"./data.csv"
        rgb_image_paths: 包含所有图片路径的列表，默认为空列表
    '''
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 创建一个新列，并初始化为0
    df['Replicate'] = 0

    # 遍历每一行
    for index, row in df.iterrows():
        img_path = "image_"+row['img_path'].split('/')[-1]
        
        # 如果图片路径不在rgb_image_paths列表中，则将Replicate列设为1
        if img_path not in rgb_image_paths:
            df.at[index, 'Replicate'] = 1
        else:
            df.at[index, 'Replicate'] = 0

    # 保存修改后的CSV文件
    df.to_csv(csv_path, index=False)

    count = 0
    # 统计多少行的['Replicate'] == 0
    for index, row in df.iterrows():
        if row['Replicate'] == 0:
            count += 1
    
    if count == len(rgb_image_paths):
        print("CSV文件修改成功！")


if __name__ == "__main__":
    modify_csv("./data copy.csv", rgb_image_paths=filter())

    