import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 讀取篩選後的 ESG 資料
df = pd.read_csv('vpesg4k_filtered.csv')

# 2. 定義分層抽樣基準
# 考量公司名稱、承諾狀態、證據狀態、證據品質、驗證時間等五項變數
tasks = ['promise_status', 'evidence_status', 'evidence_quality', 'verification_timeline']
stratify_cols = ['company'] + tasks

# 3. 預處理缺失值以生成一致的分層鍵
for col in stratify_cols:
    df[col] = df[col].fillna('N/A')

# 建立綜合分層鍵 (Stratification Key)
df['stratify_key'] = df[stratify_cols].astype(str).agg('_'.join, axis=1)

# 4. 處理小樣本類別以滿足分層抽樣之條件
# 針對僅包含單一樣本之類別進行歸類，避免 train_test_split 執行錯誤
counts = df['stratify_key'].value_counts()
df['safe_stratify_key'] = df['stratify_key'].apply(lambda x: x if counts[x] >= 2 else 'Other_Group')

# 5. 執行 50/50 平均分割 (各取 2000 筆)
train_df, test_df = train_test_split(
    df, 
    train_size=2000, 
    test_size=2000, 
    stratify=df['safe_stratify_key'], 
    random_state=42
)

# 移除過渡用之輔助欄位
train_df = train_df.drop(columns=['stratify_key', 'safe_stratify_key']).copy()
test_df = test_df.drop(columns=['stratify_key', 'safe_stratify_key']).copy()

# 6. 重新編碼 i_id
# 訓練集編號區間為 10001 至 12000
train_df['i_id'] = ['1' + str(i).zfill(4) for i in range(1, 2001)]
# 測試集編號區間為 12001 至 14000
test_df['i_id'] = ['1' + str(i).zfill(4) for i in range(2001, 4001)]

# 7. 儲存最終結果
# 使用 utf-8-sig 編碼以支援 Excel 正確顯示繁體中文
train_df.to_csv('vpesg4k_train_2000.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('vpesg4k_test_2000.csv', index=False, encoding='utf-8-sig')

print("資料集切割與重編碼完成。")
print(f"訓練集筆數：{len(train_df)}，測試集筆數：{len(test_df)}")