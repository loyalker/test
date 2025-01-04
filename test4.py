import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import time

# 切换到数据集路径
os.chdir('/home/datasets/img_data')
# 读取训练数据集
df1 = pd.read_csv('/home/datasets/train.csv')
x_train_list = []
y_train_list = []
for index, row in df1.iterrows():
    file_name = row['file_name']
    x_train = np.load(file_name)  # 加载.npy文件
    x_train_tensor = torch.from_numpy(x_train).float().unsqueeze(0).unsqueeze(0)  # 转换为张量并添加批次和通道维度
    x_train_list.append(x_train_tensor)
    y_train_list.append(row['label'])  # 假设标签在同一行的'label'列

# 将列表转换为张量
x_train_tensor = torch.cat(x_train_list, dim=0)  # 沿着批次维度合并张量
y_train_tensor = torch.tensor(y_train_list, dtype=torch.float)  # 将标签列表转换为张量

# 创建TensorDataset
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
df_val = pd.read_csv('/home/datasets/valid.csv')  
x_val_list = []
y_val_list = []
for index, row in df_val.iterrows():
    file_name = row['file_name']
    x_val = np.load(file_name)  # 加载.npy文件
    x_val_tensor = torch.from_numpy(x_val).float().unsqueeze(0).unsqueeze(0)  # 转换为张量并添加批次和通道维度
    x_val_list.append(x_val_tensor)
    y_val_list.append(row['label'])  # 假设标签在同一行的'label'列

# 将列表转换为张量
x_val_tensor = torch.cat(x_val_list, dim=0)  # 沿着批次维度合并张量
y_val_tensor = torch.tensor(y_val_list, dtype=torch.float)  # 将标签列表转换为张量
# 创建TensorDataset
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

# 创建DataLoader
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证集通常不需要打乱顺序

# 定义模型
class MRINet(nn.Module):
    def __init__(self):
        super(MRINet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),  # 假设输入数据是单通道
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(16),
            nn.Dropout(0.2),
            # 添加更多层...
        )
        self.mlp = nn.Sequential(
            nn.Linear(16*91*109*91, 128),  # 根据骨干网络的输出调整输入特征数
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 展平特征
        x = self.mlp(x)
        return x.squeeze(1)

# 初始化模型
model = MRINet()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(model, train_loader, epochs=10):
    best_acc = 0.0 
    acc_list = []  # 在函数开始处定义并初始化acc_list
    for epoch in range(epochs):
        start_time = time.time()  # 在每个epoch开始时定义start_time
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}]')
 # 保存模型
        checkpoint_dir = '/home/.ipynb_checkpoints'
        torch.save(model.state_dict(), f'{checkpoint_dir}/checkpoint_{epoch+1}.pth')
        
        # 简单验证模型
        model.eval()
        
        train_acc, val_acc = validate_model(model, train_loader, val_loader)  # 假设val_loader已定义
        
        # 记录本轮次训练的用时
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{epochs}], Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s')
        
        # 更新最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
        
        # 保存最佳模型
        torch.save(best_model_wts, f'{checkpoint_dir}/model_best.pth')
        
        # 提前停止机制
        acc_list.append(val_acc)
        if len(acc_list) > 10 and np.std(acc_list[-10:]) < 0.0003:
            print('Early stopping!')
            break
        
        # 调整学习率
        if epoch >= epochs * 1/3 and epoch < epochs * 2/3:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.2 * param_group['lr']
        
        start_time = time.time()

# 验证模型
def validate_model(model, train_loader, val_loader):
    model.eval()
    train_correct = 0
    train_total = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, target in train_loader:
            outputs = model(data)
            # 将输出转换为三分类标签
            outputs = torch.sigmoid(outputs)  # 将输出转换为概率值
            predicted = (outputs < 0.5).long() * 0 + (((outputs >= 0.5) & (outputs < 1.5)).long() * 1) + (outputs >= 1.5).long() * 2
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        for data, target in val_loader:
            outputs = model(data)
            # 将输出转换为三分类标签
            outputs = torch.sigmoid(outputs)  # 将输出转换为概率值
            predicted = (outputs < 0.5).long() * 0 + (((outputs >= 0.5) & (outputs < 1.5)).long() * 1) + (outputs >= 1.5).long() * 2
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    return train_acc, val_acc
# 开始训练
train_model(model, train_loader)