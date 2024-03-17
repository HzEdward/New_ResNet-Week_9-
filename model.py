import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import sys
import os
from PIL import Image
from dataloader import *

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
* 为了使用GPU，需要将模型和数据转移到GPU上，设置GPU.device("cuda:0"), 否则运行效果会比较慢
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

        for inputs, labels in dataloaders['train']:
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

# write a valid function to test the model
def valid_model(model, dataloaders, criterion):
    print("Validation started!")
    model.eval()
    running_loss = 0
    running_corrects = 0
    # input 即 shape
    for inputs, labels in dataloaders['val']:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders['val'].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)
    
    print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print("Validation finished!")


def test_model(model, dataloaders):
    #* Directly be given a folder, not validation but test
    #* the model would automatically output each file's prediction
    model.eval()
    for inputs, folder_names in dataloaders['val']:
        print("input.shape:", inputs.shape)
        print("folder_name:", folder_names)
        outputs = model(inputs)
        print("outputs:", outputs)
        _, preds = torch.max(outputs, 1)
        print("-------------------")
        for folder_name, pred in zip(folder_names, preds):
            print(f"File: {folder_name}, Prediction: {pred}")
    print("Validation finished!")

def resume_training(model, optimizer, checkpoint, dataloaders, criterion, num_epochs=25):
    '''
        the function using the checkpoint to resume training
    '''
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")
    print("Training started!")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
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
        
    print("Training finished!")

def test_model_checkpoint(model, dataloaders, checkpoint):
    '''
        the function to test the model on the testset by using the checkpoint
    '''
    model.load_state_dict(checkpoint['model'])
    model.eval()
    running_corrects = 0

    for inputs, labels in dataloaders['val']:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)
    print(f'Test Acc: {epoch_acc:.4f}')
    print("Testing finished!")

def create_checkpoint(model, optimizer, epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    #save checkpoint inside logs folder
    torch.save(checkpoint, 'logs/checkpoint.pth')
    return checkpoint

if __name__ == "__main__":
    # change cpu to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model, criterion, optimizer = initialize_model()
    dataloaders = get_dataloaders()
    # train_model(model, dataloaders, criterion, optimizer)
    test_model(model, dataloaders)

    

        



    


    
    

