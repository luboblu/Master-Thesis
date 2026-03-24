import json
import os
import csv
from sklearn.metrics import f1_score, classification_report

def run_evaluation():
    # --- 路徑與參數配置 ---
    # 測試集路徑
    test_dataset_path = '/Users/lubob/Desktop/Master-Thesis/dataset/Chinese_test.json'
    
    # 輸出資料夾名稱
    output_dir = 'GPT5.4_ML-Promise_Chinese_RAG_Evaluation_Results'
    
    # 建立輸出資料夾
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立輸出資料夾：{output_dir}")

    # 定義待評估的檔案及其對應的標籤欄位
    tasks = {
        '/Users/lubob/Desktop/Master-Thesis/results/GPT5.4/ML-Promise/Chinese/RAG/gpt-5.4_rag_balanced_promise_status_predictions.jsonl': 'promise_status',
        '/Users/lubob/Desktop/Master-Thesis/results/GPT5.4/ML-Promise/Chinese/RAG/gpt-5.4_rag_balanced_verification_timeline_predictions.jsonl': 'verification_timeline',
        '/Users/lubob/Desktop/Master-Thesis/results/GPT5.4/ML-Promise/Chinese/RAG/gpt-5.4_rag_balanced_evidence_status_predictions.jsonl': 'evidence_status',
        '/Users/lubob/Desktop/Master-Thesis/results/GPT5.4/ML-Promise/Chinese/RAG/gpt-5.4_rag_balanced_evidence_quality_predictions.jsonl': 'evidence_quality'
    }

    # 1. 載入測試集 (Ground Truth)
    try:
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            gt_list = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：無法讀取測試集檔案 {test_dataset_path}")
        return

    gt_data = {str(item['i_id']): item for item in gt_list}
    summary_results = []

    # 2. 執行各項任務評估
    for filepath, label_key in tasks.items():
        y_true = []
        y_pred = []
        
        if not os.path.exists(filepath):
            print(f"警告：找不到檔案 {filepath}，已跳過該項評估。")
            continue

        filename = os.path.basename(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    i_id = str(record.get('i_id'))
                    # 提取模型預測值
                    prediction = record.get('pred', {}).get(label_key)
                    
                    if i_id in gt_data and prediction is not None:
                        true_label = str(gt_data[i_id].get(label_key))
                        y_true.append(true_label)
                        y_pred.append(str(prediction))
                except json.JSONDecodeError:
                    continue
        
        if y_true:
            # 計算 Macro F1 分數
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            report = classification_report(y_true, y_pred)
            
            # 儲存摘要數據
            summary_results.append({
                'Task': label_key,
                'Sample_Size': len(y_true),
                'Macro_F1': round(macro_f1, 4)
            })

            # 產生各別任務的詳細報告文字檔，存於指定資料夾
            report_filename = os.path.join(output_dir, f"report_{label_key}.txt")
            with open(report_filename, 'w', encoding='utf-8') as rf:
                rf.write(f"Task: {label_key}\nSource: {filename}\n")
                rf.write(f"Sample Size: {len(y_true)}\n")
                rf.write(f"Macro F1 Score: {macro_f1:.4f}\n")
                rf.write("=" * 40 + "\n")
                rf.write(report)
            
            print(f"已完成 {label_key} 評估，詳細報告存至：{report_filename}")

    # 3. 匯出彙整結果至 CSV 檔案，存於指定資料夾
    if summary_results:
        output_csv = os.path.join(output_dir, 'GPT5.4_ML-Promise_Chinese_RAG_evaluation_summary.csv')
        keys = ['Task', 'Sample_Size', 'Macro_F1']
        
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(summary_results)
        
        print(f"\n所有任務評估作業已完成。")
        print(f"彙整總表路徑：{output_csv}")
    else:
        print("未產生任何評估結果，請確認檔案路徑與內容是否正確。")

if __name__ == "__main__":
    run_evaluation()