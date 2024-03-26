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
        data = pd.read_csv(os.path.join(self.root_dir, 'data copy.csv'))
        count = 0

        for index, row in data.iterrows():
            if row['blacklisted'] == 0 and row["Replicate"] == 1:
                image_path = os.path.join(self.root_dir, row['img_path'])
                label_path = os.path.join(self.root_dir, row['lbl_path'])
                self.data.append((image_path, label_path))
            else:
                pass

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
            元组: 一个包含图像和标签的元组。
        """
        image_path, label_path = self.data[idx]
        image_full_path = os.path.join('../segmentation/', image_path[2:])
        mask_full_path = os.path.join('../segmentation/', label_path[2:])

        #* 可以省略，只是为了检查路径是否正确
        # print("image_path:", image_full_path)
        # print("label_path:", mask_full_path)
        # e.g. image_path: ../segmentation/Video01/Images/Video1_frame000090.png
        #      label_path: ../segmentation/Video01/Labels/Video1_frame000090.png

        # Check if the image and label exist
        if not os.path.exists(image_full_path): 
            print("The image does not exist:", image_full_path)
            sys.exit(1)
        else:
            pass

        if not os.path.exists(mask_full_path):
            print("The label does not exist:", mask_full_path)
            sys.exit(1)
        else:
            pass

        # 将image和label读取为PIL格式
        image = Image.open(image_full_path).convert("RGB")
        mask = Image.open(mask_full_path).convert("L")

        # print(f"image_full_path:{image_full_path}, mask_full_path:{mask_full_path}")

        if self.transform:
            image = self.transform(image)
        if self.transform_segmentation:
            mask = self.transform_segmentation(mask)

        input = torch.cat([image, mask], dim=0)
    
        return input, image_full_path
 
def get_dataloaders():
    """
    Get the testing dataloaders.

    Returns:
        tuple: A tuple containing the testing dataloaders.
    """
    # Define the transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])
    transform_segmentation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.229])
    ])

    # Create the dataset
    test_dataset = SegmentationDataset(root_dir='./', transform=transform, transform_segmentation=transform_segmentation)

    # Create the dataloader
    test_loader= DataLoader(test_dataset, batch_size=1, shuffle=False)

    return {
        "test": test_loader
    }

if __name__ == '__main__':
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   dataloader = get_dataloaders()
   for i, (input, image_full_path) in enumerate(dataloader['test']):
       print("=====================================")
       print("img_path:", image_full_path)
       if i == 10:
           break

'''
Checkpoint loaded!
the testing dataset has 4275 images
Total blacklisted images: 650/4275
so percentage is 15.2%

Testing finished!

650 more blacklisted images are detected, which is consistent with the csv file
'''