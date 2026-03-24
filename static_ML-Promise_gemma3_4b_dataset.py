import json
import os
import csv
from sklearn.metrics import f1_score, classification_report

def run_evaluation():
    # 1. 配置檔案路徑與輸出參數
    test_dataset_path = '/Users/lubob/Desktop/Master-Thesis/dataset/Chinese_test.json'
    output_dir = 'Gemma3_4b_ML-Promise_Chinese_RAG_Evaluation_Results'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已成功建立輸出資料夾：{output_dir}")

    # 定義待評估的檔案
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
            # 使用 enumerate 追蹤行號，以防 idx 欄位為 N/A
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    
                    # 邏輯 A：嘗試從檔案提取 idx，若為 "N/A" 則使用當前行號
                    file_idx = record.get('idx')
                    if file_idx == "N/A" or file_idx is None:
                        current_idx = line_idx
                    else:
                        current_idx = int(file_idx)
                    
                    # 邏輯 B：兼容兩種欄位名稱 'pred' 或 'prediction'
                    pred_data = record.get('prediction') or record.get('pred', {})
                    prediction = pred_data.get(label_key)
                    
                    # 確保索引在測試集範圍內
                    if current_idx < len(gt_list) and prediction is not None:
                        true_label = str(gt_list[current_idx].get(label_key))
                        y_true.append(true_label)
                        y_pred.append(str(prediction))
                        
                except Exception as e:
                    continue
        
        if y_true:
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            report = classification_report(y_true, y_pred)
            
            summary_results.append({
                'Task': label_key,
                'Sample_Count': len(y_true),
                'Macro_F1': round(macro_f1, 4)
            })

            report_filename = os.path.join(output_dir, f"report_{label_key}.txt")
            with open(report_filename, 'w', encoding='utf-8') as rf:
                rf.write(f"Task: {label_key}\nSource: {filename}\n")
                rf.write(f"Sample Size: {len(y_true)}\n")
                rf.write(f"Macro F1 Score: {macro_f1:.4f}\n")
                rf.write("=" * 40 + "\n")
                rf.write(report)
            
            print(f"完成項目：{label_key}，樣本數：{len(y_true)}，Macro F1：{macro_f1:.4f}")

    # 4. 匯出彙整 CSV
    if summary_results:
        output_csv = os.path.join(output_dir, 'Gemma3_4b_ML-Promise_Chinese_RAG_summary.csv')
        keys = ['Task', 'Sample_Count', 'Macro_F1']
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(summary_results)
        print(f"\n所有任務評估完畢。匯總報表：{output_csv}")
    else:
        print("未產生任何有效結果，請檢查測試集與預測檔行數是否對應。")

if __name__ == "__main__":
    run_evaluation()
