#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter
import pandas as pd

# 檔案路徑
FILES = {
    "Trainset": r"C:\Users\lubob\Desktop\ITAM\dataset\PromiseEval_Trainset_French.json",
    "Testset": r"C:\Users\lubob\Desktop\ITAM\dataset\French_test.json"
}

# 子任務清單
tasks = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]

# 統計容器：{dataset: {task: Counter}}
results = {name: {task: Counter() for task in tasks} for name in FILES.keys()}

# 讀取兩個檔案並統計
for dataset, path in FILES.items():
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    for sample in data:
        for task in tasks:
            value = sample.get(task, "N/A")
            results[dataset][task][value] += 1

# 整理成 DataFrame 並計算百分比
rows = []
for dataset, task_dict in results.items():
    for task, counter in task_dict.items():
        total = sum(counter.values())
        for label, count in counter.items():
            percent = (count / total * 100) if total > 0 else 0
            rows.append([dataset, task, label, count, f"{percent:.2f}%"])

df = pd.DataFrame(rows, columns=["Dataset", "Task", "Label", "Count", "Percent"])

# 輸出成表格
print(df)

# 如果要存成 CSV
OUT_PATH = r"C:\Users\lubob\Desktop\Rockling\results\English_label_distribution_with_percent.csv"
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n已輸出結果至 {OUT_PATH}")
