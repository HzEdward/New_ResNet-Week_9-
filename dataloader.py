import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import sys
'''
 这是最原始的dataloader, 用于读取train和test数据集
 note:  
 1. __getitem__中一定要用0和1来作为返回值,否则不符合内部操作规定
 2. 为了使用GPU,需要将模型和数据转移到GPU上,设置GPU.device("cuda:0"), 否则运行效果会比较慢
 3. Final_Dataset文件夹中的数据集是最终的数据集

'''

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, transform_segmentation=None):
        self.root_dir = root_dir
        self.transform = transform
        self.transform_segmentation = transform_segmentation
        self.samples = []
        
        for label in ("blacklist", "whitelist"):
            label_dir = os.path.join(root_dir, label)
            for folder in os.listdir(label_dir):
                if folder.startswith('.'):
                    continue
                else:
                    folder_path = os.path.join(label_dir, folder)

                    for file in os.listdir(folder_path):
                        if file.startswith('image') and file.endswith('.png'):
                            rgb_image_path = os.path.join(folder_path, file)
                        elif file.startswith('label') and file.endswith('.png'):
                            segmentation_image_path = os.path.join(folder_path, file)
                    self.samples.append((rgb_image_path, segmentation_image_path, label))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, segmentation_path, label = self.samples[idx]
        rgb_image = Image.open(rgb_path).convert("RGB")
        segmentation_image = Image.open(segmentation_path).convert("L")
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
        if self.transform_segmentation:
            segmentation_image = self.transform_segmentation(segmentation_image)
    
        #! Blacklist is 0, whitelist is 1
        label = 0 if label == "whitelist" else 1
        images = torch.cat([rgb_image, segmentation_image], dim=0)

        #*rgb path is added to the return value
        return images, label, rgb_path

def get_dataloaders():
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    #* segmentation is 1 channel, so we only need to normalize it
    transform_segmentation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #TODO： check if this is the correct normalization
        transforms.Normalize(mean=[0.5], std=[0.229])
    ])
    
    train_dataset = SegmentationDataset(root_dir='../Final Dataset/train', transform=transform_rgb, transform_segmentation=transform_segmentation)
    val_dataset = SegmentationDataset(root_dir='../Final Dataset/test', transform=transform_rgb, transform_segmentation=transform_segmentation)    
    
    #! note: batch size is 24, Shuffle both are True
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)
    
    return {'train': train_loader, 'val': val_loader}

if __name__ == "__main__":
    dataloaders = get_dataloaders()    
    #* enumerate根据batch size来取数据. 
    #* torch.Size([24, 4, 224, 224]), 即24张图片，每张图片有4个channel
    print("\n")
    for i, (image, folder, img_path) in enumerate(dataloaders['val']):
        print("=====================================")
        print("image shape:", image.shape)
        print("img_path:", img_path)
        if i == 10:
            break
        
