import os
import numpy as np
import pandas as pd
import random

# 假设您的标签存储在CSV文件中，文件名为'label_info.csv'
label_info_path = '/home/processed_data.csv'
label_df = pd.read_csv(label_info_path)

# 创建文件名到标签的映射字典
file_to_label = {row['filename']: row['COG'] for index, row in label_df.iterrows()}

# 读取数据集路径
mri_path = '/home/datasets/img_data'
data_files = [os.path.join(mri_path, f) for f in os.listdir(mri_path) if f.endswith('.npy')]
data_arrays = [np.load(file) for file in data_files]

# 提取文件名
file_names = [os.path.basename(file) for file in data_files]

# 根据文件名提取标签
labels = [file_to_label.get(file_name) for file_name in file_names]

# 打乱数据顺序
random.shuffle(data_arrays)
random.shuffle(labels)

# 划分数据集
train_size = int(0.6 * len(data_arrays))
valid_size = int(0.2 * len(data_arrays))
test_size = len(data_arrays) - train_size - valid_size

train_data = data_arrays[:train_size]
valid_data = data_arrays[train_size:train_size + valid_size]
test_data = data_arrays[train_size + valid_size:]

train_labels = labels[:train_size]
valid_labels = labels[train_size:train_size + valid_size]
test_labels = labels[train_size + valid_size:]
datasets = '/home/datasets'

# 保存划分结果到csv文件
train_df = pd.DataFrame({'file_name': [os.path.basename(file) for file in data_files[:train_size]], 'label': train_labels})
train_df.to_csv(os.path.join(datasets, 'train.csv'), index=False)

valid_df = pd.DataFrame({'file_name': [os.path.basename(file) for file in data_files[train_size:train_size + valid_size]], 'label': valid_labels})
valid_df.to_csv(os.path.join(datasets, 'valid.csv'), index=False)

test_df = pd.DataFrame({'file_name': [os.path.basename(file) for file in data_files[train_size + valid_size:]], 'label': test_labels})
test_df.to_csv(os.path.join(datasets, 'test.csv'), index=False)

print("数据集划分完成！")