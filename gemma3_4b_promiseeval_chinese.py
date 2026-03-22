#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async RAG-enhanced Chinese ESG classifier (Ollama)
- Model: gemma3:4b (Optimized for 8GB VRAM)
- Fix: Global declaration SyntaxError and JSON loading logic
"""

import os, json, time, pathlib, re, sys
import numpy as np
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.asyncio import tqdm

# ================ 配置區域 ================
MODEL = "gemma3:4b"
TEMPERATURE = 0
MAX_RETRIES = 3

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 60    

CUDA_VISIBLE_DEVICES = "0"
GPU_LAYERS = 99        
GPU_MEMORY_FRACTION = 0.7 

MAX_CONCURRENT_REQUESTS = 10 
REQUESTS_PER_MINUTE = 500
BATCH_SIZE = 20

# 設定路徑
BASE_PATH = pathlib.Path(r"C:\Users\lubob\Desktop\master thesis")
TEST_PATH = BASE_PATH / "dataset" / "Chinese_test.json"
TRAIN_PATH = BASE_PATH / "dataset" / "PromiseEval_Trainset_Chinese.json"
OUT_DIR = BASE_PATH / "results" / "Gemma3_4b" / "NORAG"

USE_RAG = False  # 全域變數
USE_BALANCED_RAG = True
NUM_EXAMPLES = 6
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
# ========================================

load_dotenv()

def get_sample_content(sample: Dict[str, Any]) -> str:
    """提取文本內容：promise_string > evidence_string > text"""
    p_text = sample.get("promise_string", "")
    e_text = sample.get("evidence_string", "")
    content = p_text if p_text and p_text != "N/A" else e_text
    return content if content and content != "N/A" else sample.get("text", "")

class RAGRetriever:
    def __init__(self, embedding_model_name: str):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.train_samples: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None

    def build_index(self, train_samples: List[Dict[str, Any]]):
        self.train_samples = train_samples
        texts = [get_sample_content(s) for s in train_samples]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"[系統] RAG 索引建置完成，樣本數：{len(train_samples)}")

    def retrieve_balanced(self, query: str, task: str, k: int = 6) -> List[Dict[str, Any]]:
        if self.index is None: return []
        q = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, 20)
        
        candidates = []
        for i, idx in enumerate(idxs[0]):
            if idx < len(self.train_samples):
                item = self.train_samples[idx].copy()
                item["similarity"] = float(scores[0][i])
                candidates.append(item)
            
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for c in candidates:
            label = str(c.get(task, "N/A"))
            by_label.setdefault(label, []).append(c)
        
        selected = []
        labels = list(by_label.keys())
        while len(selected) < k and labels:
            for l in labels[:]:
                if by_label[l]:
                    selected.append(by_label[l].pop(0))
                else:
                    labels.remove(l)
                if len(selected) >= k: break
        return selected

def get_task_config(task: str) -> Dict[str, Any]:
    prompts = {
        "promise_status": ("['Yes', 'No']", "expressed concrete commitment"),
        "evidence_status": ("['Yes', 'No']", "verifiable evidence provided"),
        "evidence_quality": ("['Clear', 'Not Clear', 'Misleading', 'N/A']", "specificity and auditability"),
        "verification_timeline": ("['Already', 'Less than 2 years', '2 to 5 years', 'More than 5 years', 'N/A']", "explicit timeline")
    }
    enums, guide = prompts[task]
    sys_prompt = f"""You are an ESG inspector. Classify the input into "{task}".
Values MUST be one of {enums}. Focus on {guide}.
Return STRICT JSON format: {{"{task}": "value"}}

Examples:
{{examples}}"""
    return {"enums": eval(enums), "prompt": sys_prompt}

async def call_ollama(prompt: str) -> Dict[str, Any]:
    payload = {
        "model": MODEL, 
        "prompt": prompt, 
        "format": "json", 
        "stream": False, 
        "options": {"temperature": TEMPERATURE}
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT) as resp:
                if resp.status == 200:
                    res = await resp.json()
                    return json.loads(res.get("response", "{}"))
    except:
        return {"error": "API_CALL_FAILED"}
    return {}

async def main():
    # 修正：global 宣告必須放在函數的最開始
    global USE_RAG
    
    print(f"[啟動] 中文 ESG 分類任務 | 模型：{MODEL}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def load_json_data(path: pathlib.Path, key: str) -> List[Any]:
        if not path.exists():
            raise FileNotFoundError(f"找不到檔案: {path}")
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data.get(key, [])
            return data

    # 載入測試集
    try:
        test_samples = load_json_data(TEST_PATH, "chinese_test")
        print(f"[數據] 測試集載入成功，共 {len(test_samples)} 筆")
    except Exception as e:
        print(f"[錯誤] 測試集讀取失敗: {e}")
        return

    # 處理 RAG
    retriever = None
    if USE_RAG:
        try:
            retriever = RAGRetriever(EMBEDDING_MODEL)
            train_samples = load_json_data(TRAIN_PATH, "chinese_train")
            retriever.build_index(train_samples)
        except Exception as e:
            print(f"[警告] RAG 初始化失敗 ({e})，切換至 Zero-shot")
            USE_RAG = False

    tasks = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_item(item, task, cfg, retriever):
        async with semaphore:
            content = get_sample_content(item)
            ex_text = ""
            if USE_RAG and retriever:
                examples = retriever.retrieve_balanced(content, task, NUM_EXAMPLES)
                ex_text = "\n\n".join([f"Input: {get_sample_content(ex)}\nOutput: {{\"{task}\": \"{ex.get(task)}\"}}" for ex in examples])
            
            full_prompt = cfg["prompt"].replace("{examples}", ex_text) + f"\n\nInput: {content}\nOutput:"
            prediction = await call_ollama(full_prompt)
            # 保存索引以供後續對照
            return {"idx": item.get("idx", "N/A"), "input": content, "prediction": prediction}

    for task in tasks:
        cfg = get_task_config(task)
        print(f"\n[任務進行中] {task}")
        
        task_results = []
        # 分批執行以節省記憶體並顯示進度
        for i in range(0, len(test_samples), BATCH_SIZE):
            batch = test_samples[i:i+BATCH_SIZE]
            batch_tasks = [process_item(item, task, cfg, retriever) for item in batch]
            responses = await tqdm.gather(*batch_tasks, desc=f"批次 {i//BATCH_SIZE + 1}")
            task_results.extend(responses)

        out_path = OUT_DIR / f"chinese_gemma4b_{task}.jsonl"
        with open(out_path, "w", encoding="utf-8") as wf:
            for r in task_results:
                wf.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[完成] 結果已儲存：{out_path}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[中斷] 任務停止。")