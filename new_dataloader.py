'''
we have to create a new dataloader that created not in the form of dictionary not in real folder
input: a very bit folder in the following organization

vedio_folder(from 1 to 22):
    label folder: with labels, e.g.Video03/Labels/Video3_frame003550.png
    image folder: with images, e.g.Video03/Images/Video3_frame003550.png

csv file:
    image_path, label_path, blacklisted
    Video03/Images/Video3_frame003550.png, Video03/Labels/Video3_frame003550.png, 0
    Video03/Images/Video3_frame003550.png, Video03/Labels/Video3_frame003550.png, 1
    note: the image_path and label_path should be the same for the same frame
    Blacklisted is 0 or 1, 0 for whitelist, 1 for blacklist

all according to the csv file

output:
    organized in the form of dictionary:
{
    "train": 
        {
        "whitelist": image_path, label_path,
        "blacklist": imgae_path, label_path
        },
    "test": 
        {
        "whitelist": image_path, label_path,
        "blacklist": imgae_path, label_path
        }
}
'''
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import sys

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, transform_segmentation=None):
        self.root_dir = root_dir
        self.transform = transform # rgb image transformation
        self.transform_segmentation = transform_segmentation # segmentation image transformation
        self.data = self._load_data()

    def _load_data(self):
        # Load the CSV file
        csv_file = os.path.join(self.root_dir, 'data.csv')
        data = pd.read_csv(csv_file)

        # Initialize the dictionary to store the organized data
        organized_data = {'train': {'whitelist': [], 'blacklist': []}, 'test': {'whitelist': [], 'blacklist': []}}


        # Iterate over the rows of the CSV file
        for index, row in data.iterrows():
            image_path = os.path.join(self.root_dir, row['image_path'])
            print("image_path:", image_path)
            label_path = os.path.join(self.root_dir, row['label_path'])
            print("label_path:", label_path)
            sys.exit()
            blacklisted = row['blacklisted']

            # Determine the split (train or test) and the list (whitelist or blacklist)
            split = 'train' if index % 5 != 0 else 'test'
            list_name = 'whitelist' if blacklisted == 0 else 'blacklist'

            # Add the image and label paths to the corresponding list
            organized_data[split][list_name].append((image_path, label_path))

        return organized_data

    def __len__(self):
        return len(self.data['train']['whitelist']) + len(self.data['train']['blacklist'])

    def __getitem__(self, index):
        # Determine the split (train or test) and the list (whitelist or blacklist)
        split = 'train' if index < len(self.data['train']['whitelist']) else 'test'
        list_name = 'whitelist' if index < len(self.data['train']['whitelist']) else 'blacklist'

        # Get the image and label paths
        image_path, label_path = self.data[split][list_name][index % len(self.data[split][list_name])]

        # Load the image and label
        image = Image.open(image_path)
        label = Image.open(label_path)

        # Apply the transformations
        if self.transform:
            image = self.transform(image)
        if self.transform_segmentation:
            label = self.transform_segmentation(label)

        return image, label


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
    # train_dataset = SegmentationDataset_train(root_dir='./mock_dataset/train', transform=transform_rgb, transform_segmentation=transform_segmentation)
    # val_dataset = SegmentationDataset_test(root_dir='./mock_dataset/test', transform=transform_rgb, transform_segmentation=transform_segmentation)
    
    train_dataset = SegmentationDataset(root_dir='./mock_dataset_4/train', transform=transform_rgb, transform_segmentation=transform_segmentation)
    val_dataset = SegmentationDataset(root_dir='./mock_dataset_4/test', transform=transform_rgb, transform_segmentation=transform_segmentation)    
    
    #! note: batch size is 24
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=True)
    
    return {'train': train_loader, 'val': val_loader}     


if __name__ == '__main__':
    get_dataloaders()