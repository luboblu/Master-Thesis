import os
import warnings
os.environ["TRANSFORMERS_NO_AD_HOC_PR_CHECK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import json
import torch
import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import Dataset

# ================= 1. 路徑與任務配置 =================
TRAIN_PATH = r"/home/imntpu/Desktop/Master-Thesis/dataset/vpesg4k_train_2000.json"
TEST_PATH  = r"/home/imntpu/Desktop/Master-Thesis/dataset/vpesg4k_test_2000.json"
OUTPUT_DIR = r"/home/imntpu/Desktop/Master-Thesis/results/Roberta_ESG_Baseline"

TASK_CONFIGS = {
    "promise_status":        {"Yes": 0, "No": 1},
    "evidence_status":       {"Yes": 0, "No": 1, "N/A": 2},
    "evidence_quality":      {"Clear": 0, "Not Clear": 1, "Misleading": 2, "N/A": 3},
    "verification_timeline": {
        "already": 0,
        "within_2_years": 1,
        "between_2_and_5_years": 2,
        "longer_than_5_years": 3,
        "N/A": 4
    }
}

CONFIG = {
    "base_model":    "hfl/chinese-roberta-wwm-ext",
    "max_length":    256,
    "batch_size":    16,
    "epochs":        5,
    "learning_rate": 2e-5
}

# ================= 2. GPU 資訊 =================
def print_gpu_info():
    if torch.cuda.is_available():
        device_count   = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name    = torch.cuda.get_device_name(current_device)
        device_props   = torch.cuda.get_device_properties(current_device)
        print(f"\n[系統訊息] 檢測到可用 GPU 數量：{device_count}")
        print(f"[系統訊息] 當前正在使用的 GPU 編號：{current_device}")
        print(f"[系統訊息] GPU 型號：{device_name}")
        print(f"[系統訊息] 顯存總量：{device_props.total_memory / 1024**3:.2f} GB\n")
    else:
        print("\n[系統訊息] 未檢測到可用 GPU，將切換至 CPU 模式執行。\n")

# ================= 3. 主流程 =================
def run_baseline():
    # 載入資料
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_json = json.load(f)
        train_raw = train_json["data"] if "data" in train_json else train_json

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_json = json.load(f)
        test_raw = test_json["data"] if "data" in test_json else test_json

    tokenizer = BertTokenizer.from_pretrained(CONFIG["base_model"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for task_name, label_map in TASK_CONFIGS.items():
        print(f"\n>>> 任務開始：{task_name}")

        # 直接使用原始訓練資料，不做擴增
        train_list = [
            {"text": d["data"], "label": label_map[d[task_name]]}
            for d in train_raw
            if task_name in d and d[task_name] in label_map
        ]
        print(f"    訓練樣本數：{len(train_list)}")

        train_ds = Dataset.from_list(train_list).map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=CONFIG["max_length"]
            ),
            batched=True
        )

        # 微調 RoBERTa
        model = BertForSequenceClassification.from_pretrained(
            CONFIG["base_model"], num_labels=len(label_map)
        )
        model_path = os.path.join(OUTPUT_DIR, f"model_{task_name}")

        training_args = TrainingArguments(
            output_dir=model_path,
            per_device_train_batch_size=CONFIG["batch_size"],
            num_train_epochs=CONFIG["epochs"],
            learning_rate=CONFIG["learning_rate"],
            save_strategy="no",
            report_to="none"
        )

        Trainer(model=model, args=training_args, train_dataset=train_ds).train()
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        # 推論
        device    = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
        inv_map   = {v: k for k, v in label_map.items()}

        inference_results = []
        for item in test_raw:
            pred      = classifier(item["data"], truncation=True, max_length=512)[0]
            label_idx = int(pred["label"].split("_")[-1])
            inference_results.append({
                "i_id": item["i_id"],
                "data": item["data"],
                "pred": inv_map[label_idx]
            })

        # 儲存結果
        out_file = os.path.join(OUTPUT_DIR, f"vpesg_{task_name}_results.jsonl")
        with open(out_file, "w", encoding="utf-8") as wf:
            for r in inference_results:
                wf.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"--- 任務 {task_name} 完成，結果儲存至 {out_file} ---")

if __name__ == "__main__":
    print_gpu_info()
    run_baseline()
