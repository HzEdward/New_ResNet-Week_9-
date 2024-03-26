# Dataset for the recording

## Dataset Description
### Final Dataset: used for training and testing
### ../Mislabelled Attempts (week 6)/dataset: 未有进行类平衡的数据集


## Code 
1. dataloader.py: 用于加载数据集
2. model.py: 用于定义模型

3. new_dataloader.py: 用于加载数据集
4. new_model.py: 用于定义模型


data modified.csv: 储存了ResNet检测到的错误标签，这些疑似错误标签被标记为3
data copy.csv: 用于filter.py的过滤数据集，将已经出现在training dataset的数据过滤掉。于是添加了“replicate”这一个column

