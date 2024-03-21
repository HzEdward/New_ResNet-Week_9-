import pandas as pd
import numpy as np
import os

def stastics(csv_path="./data.csv"):
    # 统计有多少个黑名单
    data = pd.read_csv(csv_path)
    count=0
    non_count=0
    for index, row in data.iterrows():
        blacklisted = row['blacklisted']
        if blacklisted != 0:
            count += 1
        else:
            non_count += 1

    print("The number of blacklisted is:", count)
    print("The number of non-blacklisted is:", non_count)
    sum = count + non_count
    print("The total number of data is:", sum)
    

if __name__ == "__main__":
    stastics()

'''
The number of blacklisted is: 200
The number of non-blacklisted is: 4470

percentage of blacklisted: 4.29%

'''