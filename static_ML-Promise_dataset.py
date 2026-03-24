import json
import os
import csv
from sklearn.metrics import f1_score, classification_report

def run_evaluation():
    # 1. 配置檔案路徑與輸出參數
    # 請根據您的實際存放位置修改路徑
    test_dataset_path = '/Users/lubob/Desktop/Master-Thesis/dataset/Chinese_test.json'
    
    # 輸出資料夾名稱
    output_dir = 'Gemma3_4b_ML-Promise_Chinese_RAG_Evaluation_Results'
    
    # 建立輸出目錄
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已成功建立輸出資料夾：{output_dir}")

    # 定義待評估的檔案及其對應的標籤欄位
    # 注意：此處已將索引鍵由 i_id 更改為對應預測檔中的 idx
    tasks = {
        '/Users/lubob/Desktop/Master-Thesis/results/Gemma3_4b/ML-Promise/Chinese/RAG/chinese_gemma4b_promise_status.jsonl': 'promise_status',
        '/Users/lubob/Desktop/Master-Thesis/results/Gemma3_4b/ML-Promise/Chinese/RAG/chinese_gemma4b_verification_timeline.jsonl': 'verification_timeline',
        '/Users/lubob/Desktop/Master-Thesis/results/Gemma3_4b/ML-Promise/Chinese/RAG/chinese_gemma4b_evidence_status.jsonl': 'evidence_status',
        '/Users/lubob/Desktop/Master-Thesis/results/Gemma3_4b/ML-Promise/Chinese/RAG/chinese_gemma4b_evidence_quality.jsonl': 'evidence_quality'
    }

    # 2. 載入測試集 (Ground Truth)
    try:
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            gt_list = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：無法讀取測試集檔案 {test_dataset_path}")
        return

    # 關鍵修正：使用列表索引作為 ID
    # 因為 Chinese_test.json 是列表格式且無 i_id，我們以索引作為匹配基準
    gt_data = {str(i): item for i, item in enumerate(gt_list)}
    summary_results = []

    # 3. 遍歷各項預測任務
    for filepath, label_key in tasks.items():
        y_true = []
        y_pred = []
        
        if not os.path.exists(filepath):
            print(f"警告：找不到預測檔案 {filepath}，已跳過。")
            continue

        filename = os.path.basename(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    # 關鍵修正：從預測檔中提取 'idx'
                    idx = str(record.get('idx'))
                    prediction = record.get('pred', {}).get(label_key)
                    
                    # 比對索引是否存在於測試集中
                    if idx in gt_data and prediction is not None:
                        # 從測試集提取正確標籤
                        true_label = str(gt_data[idx].get(label_key))
                        y_true.append(true_label)
                        y_pred.append(str(prediction))
                except (json.JSONDecodeError, KeyError):
                    continue
        
        if y_true:
            # 計算指標與產生報告
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            report = classification_report(y_true, y_pred)
            
            summary_results.append({
                'Task': label_key,
                'Sample_Count': len(y_true),
                'Macro_F1': round(macro_f1, 4)
            })

            # 儲存詳細分類報告
            report_filename = os.path.join(output_dir, f"report_{label_key}.txt")
            with open(report_filename, 'w', encoding='utf-8') as rf:
                rf.write(f"Task: {label_key}\nSource: {filename}\n")
                rf.write(f"Sample Size: {len(y_true)}\n")
                rf.write(f"Macro F1 Score: {macro_f1:.4f}\n")
                rf.write("=" * 40 + "\n")
                rf.write(report)
            
            print(f"完成項目：{label_key}，Macro F1 為 {macro_f1:.4f}")

    # 4. 匯出彙整 CSV 報表
    if summary_results:
        output_csv = os.path.join(output_dir, 'Gemma3_4b_ML-Promise_Chinese_RAG_summary.csv')
        keys = ['Task', 'Sample_Count', 'Macro_F1']
        
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(summary_results)
        
        print(f"\n所有任務評估完畢。匯總報表路徑：{output_csv}")
    else:
        print("未產生任何有效結果，請確認檔案中的索引與欄位名稱是否正確。")

if __name__ == "__main__":
    run_evaluation()