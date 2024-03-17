import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import sys

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
        label = 0 if label == "whitelists" else 1
        images = torch.cat([rgb_image, segmentation_image], dim=0)
        print(images.shape)

        return images, label

# need to rewrite a dataset class since test dataset is different from training dataset
# test dataset only do not have whitelist and blacklist, all files are in the same folder
class SegmentationDataset_train(Dataset):
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
    
        #* Label: Blacklist is 0, whitelist is 1
        label = 0 if label == "whitelist" else 1
        images = torch.cat([rgb_image, segmentation_image], dim=0)

        return images, label

class SegmentationDataset_test(Dataset):
    def __init__(self, root_dir, transform=None, transform_segmentation=None):
        self.root_dir = root_dir
        self.transform = transform
        self.transform_segmentation = transform_segmentation
        self.samples = []
        
        for folder in os.listdir(root_dir):
            if folder.startswith('.'):
                continue
            else:
                folder_path = os.path.join(root_dir, folder)

                for file in os.listdir(folder_path):
                    if file.startswith('image') and file.endswith('.png'):
                        rgb_image_path = os.path.join(folder_path, file)
                    elif file.startswith('label') and file.endswith('.png'):
                        segmentation_image_path = os.path.join(folder_path, file)
                self.samples.append((rgb_image_path, segmentation_image_path, folder))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, segmentation_path, folder = self.samples[idx]
        rgb_image = Image.open(rgb_path).convert("RGB")
        segmentation_image = Image.open(segmentation_path).convert("L")
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
        if self.transform_segmentation:
            segmentation_image = self.transform_segmentation(segmentation_image)
    
        images = torch.cat([rgb_image, segmentation_image], dim=0)

        return images, folder

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
    
    #* Testing on the final dataset
    train_dataset = SegmentationDataset_train(root_dir='./mock_dataset/train', transform=transform_rgb, transform_segmentation=transform_segmentation)
    val_dataset = SegmentationDataset_test(root_dir='./mock_dataset/test', transform=transform_rgb, transform_segmentation=transform_segmentation)
    
    #! note: batch size is 24
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)
    
    return {'train': train_loader, 'val': val_loader}

if __name__ == "__main__":
    dataloaders = get_dataloaders()
    
    # # enumerate根据batch size来取数据 size = 24
    # for i, (image, labels) in enumerate(dataloaders['train']):
    #     print("image shape:", image.shape)
    #     print("labels:", labels)
    #     if i == 0:
    #         break
    #     # torch.Size([8, 4, 224, 224]), 即8张图片，每张图片有4个channel


    for i, (image, folder) in enumerate(dataloaders['val']):
        print("image shape:", image.shape)
        if i == 0:
            break
        
        # torch.Size([4, 4, 224, 224]), 即4张图片，每张图片有4个channel

