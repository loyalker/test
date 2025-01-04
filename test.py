
import pandas as pd
 

# 这里我们创建一个示例DataFrame来演示

df = pd.read_csv(r"/home/datasets/noimg_data/data.csv")
 
columns_to_drop = ['path','NC','MCI','DE','AD','PD','FTD','VD','DLB','PDD','ADD','OTHER','COG_score','ADD_score']
df = df.drop(columns=columns_to_drop)
 

# 计算每列中缺失值的比例
missing_ratios = df.isnull().mean()
 
# 筛选出缺失值比例超过50%的列名
columns_to_drop = missing_ratios[missing_ratios > 0.5].index
 
# 删除这些列
df = df.drop(columns=columns_to_drop)
# 定义计算单一值比例的函数
def calculate_single_value_ratio(series):
    value_counts = series.value_counts()
    most_common_count = value_counts.max()
    return most_common_count / len(series)
 
# 对DataFrame的每一列应用该函数
single_value_ratios = df.apply(calculate_single_value_ratio)
 
 
# 筛选出单一值比例超过85%的列名
columns_to_drop = single_value_ratios[single_value_ratios > 0.85].index
 
# 删除这些列
df = df.drop(columns=columns_to_drop)
# 识别出包含缺失值的列
columns_with_missing = df.columns[df.isnull().any()]
 

 
# 遍历包含缺失值的列，并计算均值填充缺失值
for col in columns_with_missing:
    # 如果该列原本就没有缺失值，则跳过（虽然在这个例子中不会发生，但为了通用性还是加上）
    if not df[col].isnull().any():
        continue
    # 计算非缺失值的均值
    mean_value = df[col].dropna().mean()
    # 使用均值填充缺失值
    df[col] = df[col].fillna(mean_value)
 
print(df)
# 假设df_filled是你已经处理过并且想要保存的DataFrame
import os
os.chdir('/home')
df.to_csv('/home/processed_data.csv', index=False)