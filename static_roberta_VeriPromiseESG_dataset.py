import json
import os
import csv
from sklearn.metrics import f1_score, classification_report

def run_evaluation():
    # 測試集路徑
    test_dataset_path = 'C:\\Users\\lubob\\Desktop\\master thesis\\dataset\\vpesg4k_test_2000.json'
    
    # 輸出目錄
    output_dir = 'Roberta_VeriPromiseESG_Evaluation_Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定義檔案與對應標籤
    tasks = {
        'C:\\Users\\lubob\\Desktop\\master thesis\\results\\Roberta_ESG\\vpesg_promise_status_results.jsonl': 'promise_status',
        'C:\\Users\\lubob\\Desktop\\master thesis\\results\\Roberta_ESG\\vpesg_verification_timeline_results.jsonl': 'verification_timeline',
        'C:\\Users\\lubob\\Desktop\\master thesis\\results\\Roberta_ESG\\vpesg_evidence_status_results.jsonl': 'evidence_status',
        'C:\\Users\\lubob\\Desktop\\master thesis\\results\\Roberta_ESG\\vpesg_evidence_quality_results.jsonl': 'evidence_quality'
    }

    # 1. 載入測試集 (Ground Truth)
    with open(test_dataset_path, 'r', encoding='utf-8') as f:
        gt_list = json.load(f)
    gt_data = {str(item['i_id']): item for item in gt_list}
    
    summary_results = []

    # 2. 執行評估
    for filepath, label_key in tasks.items():
        if not os.path.exists(filepath):
            continue

        y_true = []
        y_pred = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    i_id = str(record.get('i_id'))
                    pred_data = record.get('pred')
                    
                    # 修正：判斷 pred_data 是字典還是字串
                    if isinstance(pred_data, dict):
                        prediction = pred_data.get(label_key)
                    else:
                        prediction = pred_data
                    
                    if i_id in gt_data and prediction is not None:
                        true_label = str(gt_data[i_id].get(label_key))
                        y_true.append(true_label)
                        y_pred.append(str(prediction))
                except json.JSONDecodeError:
                    continue
        
        if y_true:
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            report_text = classification_report(y_true, y_pred)
            
            summary_results.append({
                'Task': label_key,
                'Sample_Size': len(y_true),
                'Macro_F1': round(macro_f1, 4)
            })

            # 儲存詳細報告
            report_path = os.path.join(output_dir, f"report_{label_key}.txt")
            with open(report_path, 'w', encoding='utf-8') as rf:
                rf.write(f"Task: {label_key}\nMacro F1: {macro_f1:.4f}\n\n{report_text}")

    # 3. 匯出彙整表
    if summary_results:
        output_csv = os.path.join(output_dir, 'Roberta_VeriPromiseESG_evaluation_summary.csv')
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['Task', 'Sample_Size', 'Macro_F1'])
            writer.writeheader()
            writer.writerows(summary_results)
        print(f"評估完成，彙整表已存至：{output_csv}")

if __name__ == "__main__":
    run_evaluation()