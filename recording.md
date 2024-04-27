# Dataset for the recording

## Dataset Description
### Final Dataset: used for training and testing
### ../Mislabelled Attempts (week 6)/dataset: 未有进行类平衡的数据集


## Code 
1. dataloader.py: For loading datasets (training set, validation set)
2. model.py: Used to define the model (training set, validation set)

3. test_dataloader.py: For loading test sets
4. test_model.py: Used to define test models

5. comparison.py: For comparing the results of the two models, TODO: There's a better way to represent mislabelled data.

## Dataset
data.csv: the data set used for training, the original data.csv
data copy.csv: Filter dataset for filter.py, filter out the data that already appears in the training dataset. Added the column "duplicate".
data modified.csv: Stores error labels detected by ResNet, which are labelled as 3, but these 3s are eventually labelled as 1.
data modified copy.csv: stores the error labels detected by ResNet, these suspected error labels are marked as 3.


## utils
1. filter.py. 
   filter: organises what data needs to be filtered out, i.e. what has already appeared in the training dataset.
   modify_csv: Used to modify the csv file to filter out data that has already appeared in the training dataset. The mislabels detected by ResNet, these suspected mislabels are marked as 3
   

2. modify_csv.py: Used to modify the csv file to filter out ResNet-detected mislabels, which are suspected to be 3.






