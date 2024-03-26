import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import sys
import os
from PIL import Image
from test_dataloader import *
from comparison import overlay_mask
import pandas as pd
import matplotlib.pyplot as plt

'''
这是模型的主入口，包含以下功能：
    1. initialize_model: 初始化模型、标准和优化器
    2. train_model: 训练模型
    3. valid_model: 验证模型
    4. resume_training: 从检查点恢复训练
    5. test_model: 在测试集上测试模型
    创建检查点(create_checkpoint): 为模型创建检查点

note: 
* the dataloader is defined in dataloader.py。在本文件中Dataset读取被注释了，实际上这是一个可以工作的例子。
* __getitem__中一定要用0和1来作为返回值,否则不符合内部操作规定

在运行程序时注意：
* 为了使用GPU,需要将模型和数据转移到GPU上,设置GPU.device("cuda:0"), 否则运行效果会比较慢
'''

class SingleInputResNet(nn.Module):
    def __init__(self):
        super(SingleInputResNet, self).__init__()
        # 加载预训练的 ResNet-50 模型
        self.resnet = models.resnet50(pretrained=True)
        
        # 冻结模型的所有参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 修改 ResNet-50 的输入通道数
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 获取 ResNet-50 的全连接层输入特征数
        num_ftrs = self.resnet.fc.in_features
        
        # 定义新的全连接层，输出维度为 2，即两个类别
        self.fc = nn.Linear(num_ftrs, 2)  # 输入特征数为 num_ftrs，输出维度为 2

    def forward(self, input):
        # 传递输入图像到 ResNet-50 模型，直到最后一个卷积层的输出
        x = self.resnet.conv1(input)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # 应用全局平均池化
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)  # 展平特征向量
        
        # 使用新的全连接层进行分类
        output = self.fc(x)
        return output

def initialize_model():
    model = SingleInputResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, criterion, optimizer

def train_model(model, dataloaders, criterion, optimizer, num_epochs=30):
    print("Training started!")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        running_corrects = 0

        for inputs, labels, rgb_path in dataloaders['train']:
            optimizer.zero_grad() 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        
        print(f'Epoch {epoch}/{num_epochs - 1} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        if epoch_acc == 1:
            break
    print("Training finished!")

    # save the checkpoint later could be used in the test.
    create_checkpoint(model, optimizer, epoch)

def create_checkpoint(model, optimizer, epoch):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, "checkpoint.pth")
    print("Checkpoint created!")

# write a valid function to test the model
def valid_model(model, dataloaders, criterion, num_epochs=1, checkpoint_loaded=False):
    if checkpoint_loaded:
        checkpoint = torch.load("checkpoint.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print("Checkpoint loaded!")
    else:
        pass

    print("Validation started!")
    print("=====================================")

    for epoch in range(num_epochs):
        model.eval()  # 确保模型处于评估模式
        running_loss = 0
        running_corrects = 0

        # 在验证阶段，我们不需要更新参数
        with torch.no_grad():  # 禁用梯度计算
            for inputs, labels, rgb_path in dataloaders['val']:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                print(f"length of rgb_path:{len(rgb_path)}, rgb_path:{rgb_path}")
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['val'].dataset)
        print("len(dataloaders['val'].dataset):", len(dataloaders['val'].dataset))
        print("=====================================")
        epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)
        
        print(f'Epoch {epoch}/{num_epochs - 1} Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print("Validation finished!")

def test_model(model, dataloaders, checkpoint_loaded=False):
    model.eval()

    if checkpoint_loaded:
        checkpoint = torch.load("checkpoint.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print("Checkpoint loaded!")
    else:
        pass
    
    count_blacklisted = 0
    with torch.no_grad():
        for input, image_full_path in dataloaders['test']:
            
            outputs = model(input)
            _, preds = torch.max(outputs, 1)
            if preds == 0:
                count_blacklisted += 1 

                #* 开始处理疑似黑名单的图像的比较，将其保存到output_images文件夹中
                #TODO：以一种更好的方式处理黑名单图像，可以借鉴在CATARACT2024中的colourmap的方式
                # Assuming image_full_path is a tuple containing the path as its first element
                image_full_path = image_full_path[0]
                # turn "../segmentation/Video01/Images/Video1_frame000470.png" to "../segmentation/Video01/Labels/Video1_frame000470.png"
                label_full_path = image_full_path.replace("Images", "Labels")

                # 检查图像和标签是否存在
                if not os.path.exists(image_full_path):
                    print(f"Image not found: {image_full_path}")
                    sys.exit()
                else:
                    pass
                if not os.path.exists(label_full_path):
                    print(f"Label not found: {label_full_path}")
                    sys.exit()
                else:
                    pass
                # print(f"image_full_path:{image_full_path}")
                # print(f"label_full_path:{label_full_path}")
                composite_image = overlay_mask(image_full_path,label_full_path, alpha=0.7, colormap='viridis')
                output_folder = "output_images"
                output_filename = os.path.basename(image_full_path)
                output_path = os.path.join(output_folder, output_filename)

                os.makedirs(output_folder, exist_ok=True)
                plt.imsave(output_path, composite_image)
                # print(f"Composite image saved to {output_path}")
                #* Complete 收集黑名单叠加图像


                #* 修改csv文件
                # image_full_path: ('../segmentation/Video01/Images/Video1_frame000120.png',)
                # to Video01/Images/Video1_frame000120.png
                # print(f"image_full_path:{image_full_path}")
                image_full_path = image_full_path.split("/")[2:]
                image_full_path = "/".join(image_full_path)
                # print(f"image_full_path:{image_full_path}")

                #* 则修改该csv文件中的黑名单图像的标签
                #* 将row中的blacklist的column修改为3
                #* 读取csv文件


                csv_path = "./data modified.csv"
                csv_data = pd.read_csv(csv_path)
                for i, row in csv_data.iterrows():
                    if row["img_path"] == image_full_path:
                        # 那么这行的"blacklisted"的column要修改为3
                        csv_data.at[i, "blacklisted"] = 3
                        csv_data.to_csv(csv_path, index=False)
                        print(f"Successfully modified the csv file: {csv_path}")
                        break
                    else:
                        continue
                #* 完成csv文件的修改
            else:
                continue
    

    print(f"Total blacklisted images: {count_blacklisted}/{len(dataloaders['test'].dataset)}")            
    print("Testing finished!")
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, criterion, optimizer = initialize_model()
    dataloaders = get_dataloaders()
    test_model(model, dataloaders, checkpoint_loaded=True)

    

        



    


    
    

