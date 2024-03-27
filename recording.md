# Dataset for the recording

## Dataset Description
### Final Dataset: used for training and testing
### ../Mislabelled Attempts (week 6)/dataset: 未有进行类平衡的数据集


## Code 
1. dataloader.py: 用于加载数据集（训练集，验证集）
2. model.py: 用于定义模型（训练集，验证集）

3. test_dataloader.py: 用于加载测试集
4. test_model.py: 用于定义测试模型

5. comparison.py: 用于比较两个模型的结果, TODO: 还要用更好的方式表现出mislabelled的数据

## Dataset
data.csv: 用于训练的数据集, 最原始的data.csv
data copy.csv: 用于filter.py的过滤数据集，将已经出现在training dataset的数据过滤掉。添加了“replicate”这一个column
data modified.csv: 储存了ResNet检测到的错误标签，这些疑似错误标签被标记为3，但是这些3最后被标记为1
data modified copy.csv: 储存了ResNet检测到的错误标签，这些疑似错误标签被标记为3


## utils
1. filter.py: 
   filter: 整理出哪些数据是需要被过滤掉的，即在训练集中已经出现过
   modify_csv: 用于修改csv文件，将已经出现在training dataset的数据过滤掉。将ResNet检测到的错误标签，这些疑似错误标签被标记为3
   

2. modify_csv.py: 用于修改csv文件，将ResNet检测到的错误标签，这些疑似错误标签被标记为3





