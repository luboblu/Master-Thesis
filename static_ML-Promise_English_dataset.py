import json
import os
import csv
from sklearn.metrics import f1_score, classification_report

def run_evaluation():
    # 1. 配置路徑：請確保路徑指向您的英文實驗檔案
    test_dataset_path = 'C:\\Users\\lubob\\Desktop\\master thesis\\dataset\\English_test.json'
    output_dir = 'C:\\Users\\lubob\\Desktop\\master thesis\\statistic\\Gemma3_4b_ML-Promise_English_NORAG_Evaluation_Results'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立輸出資料夾：{output_dir}")

    # 定義英文任務檔案與對應標籤
    tasks = {
       'C:\\Users\\lubob\\Desktop\\master thesis\\results\\Gemma3_4b\\ML-Promise\\English\\NORAG\\english_ollama_promise_status_predictions.jsonl': 'promise_status',
        'C:\\Users\\lubob\\Desktop\\master thesis\\results\\Gemma3_4b\\ML-Promise\\English\\NORAG\\english_ollama_verification_timeline_predictions.jsonl': 'verification_timeline',
        'C:\\Users\\lubob\\Desktop\\master thesis\\results\\Gemma3_4b\\ML-Promise\\English\\NORAG\\english_ollama_evidence_status_predictions.jsonl': 'evidence_status',
        'C:\\Users\\lubob\\Desktop\\master thesis\\results\\Gemma3_4b\\ML-Promise\\English\\NORAG\\english_ollama_evidence_quality_predictions.jsonl': 'evidence_quality'
    }

    # 2. 載入測試集標籤 (Ground Truth)
    try:
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            gt_list = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：無法讀取測試集 {test_dataset_path}")
        return

    summary_results = []

    # 3. 執行指標計算
    for filename, label_key in tasks.items():
        y_true = []
        y_pred = []
        
        if not os.path.exists(filename):
            print(f"跳過：找不到檔案 {filename}")
            continue

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    # 依據提供格式：從 'idx' 取得索引，從 'pred' 字典取得預測值
                    idx = record.get('idx')
                    prediction = record.get('pred', {}).get(label_key)
                    
                    if idx is not None and idx < len(gt_list) and prediction is not None:
                        # 提取測試集對應索引的正確標籤
                        true_label = str(gt_list[idx].get(label_key))
                        y_true.append(true_label)
                        y_pred.append(str(prediction))
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        
        if y_true:
            # 計算 Macro F1 指標
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            report = classification_report(y_true, y_pred)
            
            summary_results.append({
                'Task': label_key,
                'Samples': len(y_true),
                'Macro_F1': round(macro_f1, 4)
            })

            # 儲存詳細分類文本報告
            report_path = os.path.join(output_dir, f"report_{label_key}.txt")
            with open(report_path, 'w', encoding='utf-8') as rf:
                rf.write(f"Task: {label_key}\nFile: {filename}\n")
                rf.write(f"Macro F1 Score: {macro_f1:.4f}\n")
                rf.write("=" * 40 + "\n")
                rf.write(report)
            
            print(f"項目 {label_key} 評估完成，Macro F1: {macro_f1:.4f}")

    # 4. 匯出彙整 CSV 報表
    if summary_results:
        csv_path = os.path.join(output_dir, 'Gemma3_4b_ML-Promise_English_NORAG_evaluation_summary.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['Task', 'Samples', 'Macro_F1'])
            writer.writeheader()
            writer.writerows(summary_results)
        print(f"\n所有評估已完成。總結報表存於：{csv_path}")

if __name__ == "__main__":
    run_evaluation()