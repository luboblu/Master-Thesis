#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate French ESG predictions for 4 subtasks
→ 指標：Precision_Macro、Recall_Macro、F1_Macro、F1_Micro（不含 Accuracy）
→ 產出 CSV 與每任務＋整體平均的長條圖
→ 檔名格式：french_<strategy>_<task>_predictions.jsonl
"""

import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import os

# ===== 字型修正區（解決中文變方塊） =====
CJK_CANDIDATES = [
    "Microsoft JhengHei", "PMingLiU", "MingLiU",     # Windows 繁中
    "SimHei", "Microsoft YaHei",                     # Windows 簡中
    "Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP",
    "Noto Serif CJK TC", "Noto Serif CJK SC",
]

FONT_FILE = None  # 若想手動指定字型檔案路徑（例如 C:\Windows\Fonts\msjh.ttc），可改這裡

def set_cjk_font():
    if FONT_FILE and os.path.exists(FONT_FILE):
        font_manager.fontManager.addfont(FONT_FILE)
        prop = font_manager.FontProperties(fname=FONT_FILE)
        rcParams["font.sans-serif"] = [prop.get_name()]
    else:
        available = set(f.name for f in font_manager.fontManager.ttflist)
        picked = None
        for name in CJK_CANDIDATES:
            if name in available:
                picked = name
                break
        if picked:
            rcParams["font.sans-serif"] = [picked]
        else:
            print("⚠ 找不到中文字型，請安裝/指定 Noto Sans CJK 或 微軟正黑體。")
    rcParams["axes.unicode_minus"] = False

set_cjk_font()
# =========================================


# ========= 路徑設定 =========
TEST_PATH = r"C:\Users\lubob\Desktop\ITAM\dataset\French_test.json"
PRED_DIR  = r"C:\Users\lubob\Desktop\ITAM\results\gpt5_rag"
OUT_DIR   = r"C:\Users\lubob\Desktop\ITAM\dataset\analysis"
OUT_CSV   = str(pathlib.Path(OUT_DIR) / "gpt5_rag_french_evaluation_summary.csv")  # ✅ 改成合法檔名

# ========= 子任務 =========
TASKS = [
    "promise_status",
    "evidence_status",
    "evidence_quality",
    "verification_timeline",
]

# ========= 檔名正則 =========
FNAME_PATTERN = re.compile(
    r'^french_(?P<strategy>[a-zA-Z0-9_]+)_(?P<task>promise_status|evidence_status|evidence_quality|verification_timeline)_predictions\.jsonl$',
    re.IGNORECASE
)

# ========= 載入 Ground Truth =========
def load_ground_truth(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    gt = defaultdict(list)
    for row in data:
        for task in TASKS:
            gt[task].append(row.get(task, None))
    print(f"[INFO] Ground Truth 樣本數: {len(data)}")
    return gt

# ========= 掃描預測檔 =========
def discover_prediction_files(pred_dir: pathlib.Path) -> Dict[str, Dict[str, pathlib.Path]]:
    mapping: Dict[str, Dict[str, pathlib.Path]] = {t: {} for t in TASKS}
    for p in pred_dir.glob("*_predictions.jsonl"):
        m = FNAME_PATTERN.match(p.name)
        if not m:
            continue
        strategy = m.group("strategy")
        task = m.group("task")
        mapping[task][strategy] = p
    return mapping

# ========= 載入預測 =========
def load_predictions(pred_path: pathlib.Path, task: str) -> List[str]:
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "pred" in rec and isinstance(rec["pred"], dict):
                    preds.append(rec["pred"].get(task))
                else:
                    preds.append(rec.get(task))
            except Exception:
                preds.append(None)
    return preds

# ========= 對齊 =========
def align_and_filter(y_true: List[str], y_pred: List[str]) -> Tuple[List[str], List[str]]:
    n = min(len(y_true), len(y_pred))
    yt, yp = [], []
    for i in range(n):
        if y_true[i] is not None and y_pred[i] is not None:
            yt.append(str(y_true[i]).strip())
            yp.append(str(y_pred[i]).strip())
    return yt, yp

# ========= 指標計算 =========
def compute_metrics(y_true: List[str], y_pred: List[str]):
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro        = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro        = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return precision_macro, recall_macro, f1_macro, f1_micro

# ========= 視覺化：單任務比較 =========
def plot_task_comparison(task: str, df_task: pd.DataFrame, out_dir: pathlib.Path):
    if df_task.empty:
        return
    x = np.arange(len(df_task))
    width = 0.20

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*width, df_task["Precision_Macro"], width=width, label="Precision_Macro", alpha=0.85)
    plt.bar(x - 0.5*width, df_task["Recall_Macro"],    width=width, label="Recall_Macro",    alpha=0.85)
    plt.bar(x + 0.5*width, df_task["F1_Macro"],        width=width, label="F1_Macro",        alpha=0.85)
    plt.bar(x + 1.5*width, df_task["F1_Micro"],        width=width, label="F1_Micro",        alpha=0.85)

    plt.xticks(x, df_task["Strategy"], rotation=15)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(f"法文ESG分類效果比較 — {task}", pad=4)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    for i in range(len(df_task)):
        plt.text(x[i] - 1.5*width, df_task.iloc[i]["Precision_Macro"] + 0.01, f'{df_task.iloc[i]["Precision_Macro"]:.3f}',
                 ha='center', va='bottom', fontsize=8)
        plt.text(x[i] - 0.5*width, df_task.iloc[i]["Recall_Macro"] + 0.01, f'{df_task.iloc[i]["Recall_Macro"]:.3f}',
                 ha='center', va='bottom', fontsize=8)
        plt.text(x[i] + 0.5*width, df_task.iloc[i]["F1_Macro"] + 0.01, f'{df_task.iloc[i]["F1_Macro"]:.3f}',
                 ha='center', va='bottom', fontsize=8)
        plt.text(x[i] + 1.5*width, df_task.iloc[i]["F1_Micro"] + 0.01, f'{df_task.iloc[i]["F1_Micro"]:.3f}',
                 ha='center', va='bottom', fontsize=8)

    out_png = out_dir / f"{task}_comparison.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"🖼️ 單任務圖已儲存: {out_png}")

# ========= 視覺化：整體平均 =========
def plot_overall_comparison(df: pd.DataFrame, out_dir: pathlib.Path):
    if df.empty:
        return
    avg_scores = df.groupby("Strategy")[["Precision_Macro", "Recall_Macro", "F1_Macro", "F1_Micro"]].mean()

    plt.figure(figsize=(10, 6))
    avg_scores.plot(kind="bar", ax=plt.gca(), alpha=0.85)
    plt.title("法文ESG分類 - 各策略平均效果（Precision/Recall/F1）", pad=4)
    plt.ylabel("平均分數")
    plt.xlabel("策略")
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)

    out_png = out_dir / "overall_comparison.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"🖼️ 整體平均圖已儲存: {out_png}")

# ========= 主程式 =========
def main():
    pred_dir = pathlib.Path(PRED_DIR)
    out_dir = pathlib.Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(TEST_PATH)
    file_map = discover_prediction_files(pred_dir)

    summary = []

    for task in TASKS:
        strategies = file_map.get(task, {})
        for strategy, pred_file in strategies.items():
            y_true = gt[task]
            y_pred = load_predictions(pred_file, task)
            yt, yp = align_and_filter(y_true, y_pred)
            if not yt:
                continue
            precision_macro, recall_macro, f1_macro, f1_micro = compute_metrics(yt, yp)
            summary.append({
                "Task": task,
                "Strategy": f"french_{strategy}",
                "Precision_Macro": precision_macro,
                "Recall_Macro": recall_macro,
                "F1_Macro": f1_macro,
                "F1_Micro": f1_micro,
                "Valid_Samples": len(yt),
                "Total_Samples": len(y_true)
            })
            print(f"{task:22s} | {strategy:18s} -> "
                  f"P_Ma={precision_macro:.3f}, R_Ma={recall_macro:.3f}, "
                  f"F1_Ma={f1_macro:.3f}, F1_Mi={f1_micro:.3f} (N={len(yt)})")

    df = pd.DataFrame(summary).sort_values(["Task", "Strategy"]).reset_index(drop=True)
    try:
        df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n📄 已輸出評估結果: {OUT_CSV}")
    except Exception as e:
        print(f"❌ 寫入 CSV 失敗：{e}")

    for task in TASKS:
        df_task = df[df["Task"] == task]
        plot_task_comparison(task, df_task, out_dir)
    plot_overall_comparison(df, out_dir)

if __name__ == "__main__":
    main()
