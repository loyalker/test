import pandas as pd
import numpy as np
 
# 假设df是你的原始数据集DataFrame
# 如果df是从CSV文件加载的，你可以使用 pd.read_csv('your_original_data.csv') 来加载它
# df = pd.read_csv('your_original_data.csv')
df = pd.read_csv('/home/processed_data.csv')
 
# 打乱数据集
df_shuffled = df.sample(frac=1).reset_index(drop=True)
 
# 计算划分索引
total_samples = len(df_shuffled)
train_size = int(0.6 * total_samples)
valid_size = int(0.2 * total_samples)
test_size = total_samples - train_size - valid_size
 
# 划分数据集
train_df = df_shuffled.iloc[:train_size]
valid_df = df_shuffled.iloc[train_size:train_size + valid_size]
test_df = df_shuffled.iloc[train_size + valid_size:]
 
datasets = '/home/datasets'
 
# 保存数据集到datasets文件夹中
train_df.to_csv(os.path.join(datasets, 'train.csv'), index=False)
valid_df.to_csv(os.path.join(datasets, 'valid.csv'), index=False)
test_df.to_csv(os.path.join(datasets, 'test.csv'), index=False)