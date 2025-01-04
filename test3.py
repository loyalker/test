import pandas as pd
from catboost import CatBoostClassifier
import os
import numpy as np
 
# 载入数据
train_data = pd.read_csv('/home/datasets/train.csv')
valid_data = pd.read_csv('/home/datasets/valid.csv')
 
# 分离特征与标签
x_train = train_data.drop('COG', axis=1)
y_train = train_data['COG']
x_valid = valid_data.drop('COG', axis=1)
y_valid = valid_data['COG']
 
# 确保数据类型和形状正确（这里只是示例，具体应根据您的数据调整）
assert isinstance(x_train, pd.DataFrame) and isinstance(x_valid, pd.DataFrame)
assert isinstance(y_train, pd.Series) and isinstance(y_valid, pd.Series)

# 配置CatBoostClassifier
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    verbose=False,
    eval_metric='Accuracy',
    use_best_model=True,
    cat_features=[1, 2]  # 确保这些列是分类特征
)
 
# 训练模型
model.fit(x_train, y_train, eval_set=(x_valid, y_valid), verbose=100)  # 可以调整verbose以查看更多训练信息
 
# 获取预测结果
train_pred_classes = model.predict(x_train)
valid_pred_classes = model.predict(x_valid)

train_pred_classes = np.argmax(train_pred_classes, axis=1)
valid_pred_classes = np.argmax(valid_pred_classes, axis=1)
# 计算准确率
train_accuracy = np.mean(train_pred_classes == y_train)

valid_accuracy = np.mean(valid_pred_classes == y_valid)
 
# 输出日志信息
print(f'Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}')
 
# 设定模型保存路径并保存模型
checkpoint_dir = '/home/.ipynb_checkpoints'  # 修改为您想要保存模型的路径
os.makedirs(checkpoint_dir, exist_ok=True)
model_path = os.path.join(checkpoint_dir, 'nonImg_model.cbm')
model.save_model(model_path)