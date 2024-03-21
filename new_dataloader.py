'''
According to the big folder containing images and labels and a csv file, we gonna create a list that in the following organization as the 
testing set of our model

csv file:
    e.g. 
    1. csv file, with the following content:
    index, image_path, label_path, blacklisted
    3103,Video17/Images/Video17_frame010480.png,Video17/Labels/Video17_frame010480.png,0
    note: Blacklisted is 0 or 1, 0 for whitelist, 1 for blacklist
    
    2. in a folder with 22 videos folder, naming like, Video01, Video02, ..., Video22
    each video folder has two subfolders, Images and Labels, and in each images folder, there are images, and in each labels folder, there are labels
    the full adress for each image is e.g. Video03/Images/Video3_frame003550.png

output: We gonna create a list that in the following organization:
    [(image_path, label_path), (image_path, label_path), ...]
    note: list of tuples, each tuple contains the image_path and label_path
    e.g [(Video03/Images/Video3_frame003550.png, Video03/Labels/Video3_frame003550.png), ...]
'''

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import sys

class SegmentationDataset(Dataset):
    """
    A custom dataset class for segmentation data.
    """

    def __init__(self, root_dir, transform=None, transform_segmentation=None):
        """
        Initialize the SegmentationDataset.

        Args:
            root_dir (str): The root directory of the dataset.
            transform (callable, optional): Optional transform to be applied to the RGB images.
            transform_segmentation (callable, optional): Optional transform to be applied to the segmentation labels.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.transform_segmentation = transform_segmentation
        self.data = []
        self.get_data() # get the data from the csv file and fill the data list

    def get_data(self):
        """
        Read the data from the CSV file and populate the data list.
        """
        data = pd.read_csv(os.path.join(self.root_dir, 'data.csv'))
        for index, row in data.iterrows():
            image_path = os.path.join(self.root_dir, row['img_path'])
            label_path = os.path.join(self.root_dir, row['lbl_path'])
            self.data.append((image_path, label_path))
        print("The length of the data is:", len(self.data))
        print("The first element of the data is:", self.data[0])

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        image_path, label_path = self.data[idx]
        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.transform:
            image = self.transform(image)
        if self.transform_segmentation:
            label = self.transform_segmentation(label)

        return image, label
 


def get_dataloaders():
    """
    Get the testing dataloaders.

    Returns:
        tuple: A tuple containing the testing dataloaders.
    """
    # Define the transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    transform_segmentation = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Create the dataset
    dataset = SegmentationDataset(root_dir='./', transform=transform, transform_segmentation=transform_segmentation)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    return dataloader


if __name__ == '__main__':
   dataset = SegmentationDataset(root_dir='./')
   dataset.get_data()    
