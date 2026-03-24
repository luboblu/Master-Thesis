import os
import warnings
# 1. 隱藏 403 Forbidden (背景討論區檢查)
os.environ["TRANSFORMERS_NO_AD_HOC_PR_CHECK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 2. 隱藏 Python 的 FutureWarning (例如 torch 那些即將棄用的警告)
warnings.filterwarnings("ignore", category=FutureWarning)

# 3. 隱藏 Transformers 的 MISSING/UNEXPECTED 表格
# 將日誌等級設為 ERROR，這樣它就只會顯示真的壞掉的東西，不會顯示警告
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
import json
import asyncio
import os
import torch
import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    pipeline
)
from datasets import Dataset
import logging

# ================= 1. 路徑與任務配置 =================
# 這裡直接導入你之前的 VeriPromiseESG 檔案路徑
TRAIN_PATH = r"/home/imntpu/Desktop/Master-Thesis/dataset/vpesg4k_train_2000.json"
TEST_PATH = r"/home/imntpu/Desktop/Master-Thesis/dataset/vpesg4k_test_2000.json"
OUTPUT_DIR = r"/home/imntpu/Desktop/Master-Thesis/results/Roberta_ESG"

# 定義四個子任務及其標籤映射 [cite: 182-184]
TASK_CONFIGS = {
    "promise_status": {"Yes": 0, "No": 1},
    "evidence_status": {"Yes": 0, "No": 1, "N/A": 2}, 
    "evidence_quality": {"Clear": 0, "Not Clear": 1, "Misleading": 2, "N/A": 3},
    "verification_timeline": {
        "already": 0, 
        "within_2_years": 1, 
        "between_2_and_5_years": 2, 
        "longer_than_5_years": 3, 
        "N/A": 4
    }
}

CONFIG = {
    "base_model": "hfl/chinese-roberta-wwm-ext",
    "openai_model": "gpt-4o", # 建議使用 GPT-4o 進行高質量數據增強
    "api_key": "",
    "max_length": 256,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 2e-5
}

# ================= 2. 數據增強 (使用 GPT 進行平衡) =================
class ESGDataAugmenter:
    def __init__(self, api_key):
        self.client = AsyncOpenAI(api_key=api_key)
        # 遵循論文改寫要求：保留術語、不引入新事實 [cite: 105-111]
        self.prompt_template = (
            "The task is to generate one alternative expression for the following ESG text "
            "while preserving its original meaning, tone, and accuracy.\n"
            "Requirements: 1. Rewrite the text. 2. Retain ESG terminology. "
            "3. No new facts. 4. Formal tone.\n\nOriginal text: {text}\nRewritten text:"
        )

    async def augment(self, text):
        try:
            resp = await self.client.chat.completions.create(
                model=CONFIG["openai_model"],
                messages=[{"role": "user", "content": self.prompt_template.format(text=text)}],
                temperature=0.7
            )
            return resp.choices[0].message.content.strip()
        except: return None

    async def balance_task_data(self, raw_data, task_name):
        """根據子任務標籤分佈進行平衡擴增 [cite: 98-100]"""
        labels = [item.get(task_name) for item in raw_data if task_name in item]
        unique, counts = np.unique(labels, return_counts=True)
        max_c = max(counts)
        
        balanced_list = list(raw_data)
        for label, count in zip(unique, counts):
            diff = max_c - count
            if diff > 0:
                candidates = [item["data"] for item in raw_data if item.get(task_name) == label]
                tasks = [self.augment(np.random.choice(candidates)) for _ in range(diff)]
                aug_res = await tqdm.gather(*tasks, desc=f"增強 {task_name}:{label}")
                balanced_list.extend([{"data": r, task_name: label} for r in aug_res if r])
        return balanced_list

# ================= 3. 整合執行流程 =================
async def run_vpesg_pipeline():
    # A. 載入原始 JSON 數據
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_json = json.load(f)
        train_raw = train_json["data"] if "data" in train_json else train_json
    
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_json = json.load(f)
        test_raw = test_json["data"] if "data" in test_json else test_json

    augmenter = ESGDataAugmenter(CONFIG["api_key"])
    tokenizer = BertTokenizer.from_pretrained(CONFIG["base_model"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 針對四個子任務循環 [cite: 305-306]
    for task_name, label_map in TASK_CONFIGS.items():
        print(f"\n>>> 任務開始：{task_name}")
        
        # 1. 數據增強與平衡 [cite: 79-82]
        balanced_data = await augmenter.balance_task_data(train_raw, task_name)
        
        # 2. 準備 Dataset
        train_list = [{"text": d["data"], "label": label_map[d[task_name]]} 
                      for d in balanced_data if task_name in d]
        train_ds = Dataset.from_list(train_list).map(
            lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=CONFIG["max_length"]), 
            batched=True
        )

        # 3. 微調 Roberta [cite: 185-190]
        model = BertForSequenceClassification.from_pretrained(CONFIG["base_model"], num_labels=len(label_map))
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

        # 4. 推論作答
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
        inv_map = {v: k for k, v in label_map.items()}
        
        inference_results = []
        for item in test_raw:
    # 增加 truncation=True，確保長度不會超過 512
            pred = classifier(item["data"], truncation=True, max_length=512)[0]
            label_idx = int(pred["label"].split("_")[-1])
            inference_results.append({
                "i_id": item["i_id"],
                "data": item["data"],
                "pred": inv_map[label_idx]
            })

        # 5. 儲存與截圖一致的 .jsonl 檔案
        out_file = os.path.join(OUTPUT_DIR, f"vpesg_{task_name}_results.jsonl")
        with open(out_file, "w", encoding="utf-8") as wf:
            for r in inference_results:
                wf.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"--- 任務 {task_name} 完成，結果儲存至 {out_file} ---")
def print_gpu_info():
    if torch.cuda.is_available():
        # 獲取可用 GPU 的數量
        device_count = torch.cuda.device_count()
        # 獲取當前正在使用的 GPU 編號
        current_device = torch.cuda.current_device()
        # 獲取該 GPU 的型號名稱
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"\n[系統訊息] 檢測到可用 GPU 數量：{device_count}")
        print(f"[系統訊息] 當前正在使用的 GPU 編號：{current_device}")
        print(f"[系統訊息] GPU 型號：{device_name}")
        # 使用 get_device_properties 來取得硬體規格資訊
        device_props = torch.cuda.get_device_properties(current_device)
        print(f"[系統訊息] 顯存總量：{device_props.total_memory / 1024**3:.2f} GB\n")
    else:
        print("\n[系統訊息] 未檢測到可用 GPU，將切換至 CPU 模式執行。\n")
if __name__ == "__main__":
    print_gpu_info()
    asyncio.run(run_vpesg_pipeline())