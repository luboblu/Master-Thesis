import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 指定資料集之本機絕對路徑
train_path = r'C:\Users\lubob\Desktop\master thesis\dataset\vpesg4k_train_2000.csv'
test_path = r'C:\Users\lubob\Desktop\master thesis\dataset\vpesg4k_test_2000.csv'

# 載入訓練集與測試集
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 定義 ESG 子任務欄位
sub_tasks = ['promise_status', 'evidence_status', 'evidence_quality', 'verification_timeline']

def plot_distribution_refined(train_series, test_series, title, filename):
    """
    優化繪圖函數：統一視覺風格與標籤對齊方式
    """
    # 計算各類別之相對頻率
    train_dist = train_series.value_counts(normalize=True).sort_index()
    test_dist = test_series.value_counts(normalize=True).sort_index()
    
    # 對齊類別索引
    all_categories = sorted(list(set(train_dist.index) | set(test_dist.index)))
    train_dist = train_dist.reindex(all_categories, fill_value=0)
    test_dist = test_dist.reindex(all_categories, fill_value=0)
    
    indices = np.arange(len(all_categories))
    width = 0.35
    
    # 建立圖表畫布
    plt.figure(figsize=(8, 6))
    
    # 繪製對照柱狀圖
    plt.bar(indices - width/2, train_dist.values, width, label='Train', color='skyblue', edgecolor='black', alpha=0.8)
    plt.bar(indices + width/2, test_dist.values, width, label='Test', color='salmon', edgecolor='black', alpha=0.8)
    
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    
    # 統一設置 X 軸標籤旋轉角度與對齊基準
    plt.xticks(indices, all_categories, rotation=45, ha='right')
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 儲存高解析度圖表
    plt.savefig(filename, dpi=300)
    plt.close()

# 1. 產出樣本數前五大公司之分佈圖
top_5_indices = train_df['company'].value_counts().nlargest(5).index
plot_distribution_refined(
    train_df[train_df['company'].isin(top_5_indices)]['company'], 
    test_df[test_df['company'].isin(top_5_indices)]['company'], 
    'Top 5 Companies Distribution', 
    'top5_companies_dist_fixed.png'
)

# 2. 依序產出四項子任務之分佈圖
for task in sub_tasks:
    plot_distribution_refined(
        train_df[task].fillna('N/A'), 
        test_df[task].fillna('N/A'), 
        f'{task} Distribution', 
        f'{task}_dist_fixed.png'
    )

print("優化後的圖表已儲存至腳本執行路徑。")