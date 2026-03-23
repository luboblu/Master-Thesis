#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非同步 RAG 增強型 ESG 分類器 (JSON 版本)
- 目標模型：gemma3:4b (透過 Ollama)
- 輸出內容：包含 i_id, data (原始文本), pred (預測結果)
"""

import os, json, time, pathlib, re, sys
import numpy as np
import asyncio
import aiohttp
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import torch
from tqdm.asyncio import tqdm

# ================ 配置參數 ================
MODEL = "gemma3:27b" 
TEMPERATURE = 0 
MAX_RETRIES = 3

# Ollama 伺服器設定
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 120

# 並行處理與速率限制
MAX_CONCURRENT_REQUESTS = 8
REQUESTS_PER_MINUTE = 300
BATCH_SIZE = 25

# 檔案路徑設定
TEST_PATH = r"/home/imntpu/Desktop/Master-Thesis/dataset/vpesg4k_test_2000.json"
TRAIN_PATH = r"/home/imntpu/Desktop/Master-Thesis/dataset/vpesg4k_train_2000.json"
OUT_DIR = r"/home/imntpu/Desktop/Master-Thesis/results/Gemma3_27b/VeriPromiseESG/NORAG"

# RAG 配置
USE_RAG = False
USE_BALANCED_RAG = True
NUM_EXAMPLES = 6
CANDIDATE_POOL_SIZE = 20
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
# ========================================

load_dotenv()

class AsyncRateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = []

    async def acquire(self):
        now = time.time()
        self.requests = [t for t in self.requests if now - t < 60]
        if len(self.requests) >= self.max_requests:
            wait = 60 - (now - self.requests[0])
            if wait > 0:
                await asyncio.sleep(wait)
                return await self.acquire()
        self.requests.append(now)

class RAGRetriever:
    def __init__(self, embedding_model_name: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"\n[系統訊息] 檢測到可用 GPU：{torch.cuda.get_device_name(0)}")
            print(f"[系統訊息] 嵌入模型將使用 CUDA 進行加速。")
        else:
            print(f"\n[系統訊息] 未檢測到 GPU，嵌入模型將使用 CPU 進行運算。")
        
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.train_samples: List[Dict[str, Any]] = []
        self.embeddings = None
        self.index = None

    def build_index(self, train_samples: List[Dict[str, Any]]):
        self.train_samples = train_samples
        texts = [str(s.get("data", "")) for s in train_samples]
        self.embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def retrieve_similar_samples(self, query_text: str, k: int = 6) -> List[Dict[str, Any]]:
        if self.index is None or not query_text: return []
        q = self.embedding_model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        out = []
        for i, idx in enumerate(idxs[0]):
            if idx < len(self.train_samples):
                item = self.train_samples[idx].copy()
                item["similarity_score"] = float(scores[0][i])
                out.append(item)
        return out

    def retrieve_balanced_samples(self, query_text: str, task: str, k: int = 6, candidate_pool: int = 20) -> List[Dict[str, Any]]:
        cand = self.retrieve_similar_samples(query_text, k=min(candidate_pool, len(self.train_samples)))
        return self._select_balanced_subset(cand, task, k)

    def _select_balanced_subset(self, candidates: List[Dict[str, Any]], task: str, k: int) -> List[Dict[str, Any]]:
        if len(candidates) <= k: return candidates
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for s in candidates:
            label = str(s.get(task, "N/A"))
            by_label.setdefault(label, []).append(s)
        selected: List[Dict[str, Any]] = []
        labels = list(by_label.keys())
        for lab in labels:
            if len(selected) < k and by_label[lab]:
                selected.append(by_label[lab].pop(0))
        while len(selected) < k:
            added = False
            for lab in labels:
                if len(selected) >= k: break
                if by_label[lab]:
                    selected.append(by_label[lab].pop(0))
                    added = True
            if not added: break
        return selected

def get_task_config(task: str) -> Dict[str, Any]:
    if task == "promise_status":
        enums = ["Yes", "No"]
        guidance = "Classify if the text contains a concrete environmental or social commitment."
    elif task == "evidence_status":
        enums = ["Yes", "No"]
        guidance = "Classify if there is clear evidence or actions supporting the commitment."
    elif task == "evidence_quality":
        enums = ["Clear", "Not Clear", "Misleading", "N/A"]
        guidance = "Assess the quality of evidence. Use 'N/A' if no evidence is present."
    elif task == "verification_timeline":
        enums = ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years", "N/A"]
        guidance = "Identify the implementation timeline mentioned in the text."
    else:
        raise ValueError(f"不支援的任務類型: {task}")

    rag_prompt = f"""You are an ESG compliance inspector.
Classify the INPUT into a single field: {task}.
Return STRICT JSON with only one key: "{task}", and the value MUST be one of {enums}.
Definition: {guidance}
Reference Examples:
{{examples}}
Return ONLY the JSON object."""

    no_rag_prompt = f"""You are an ESG compliance inspector.
Classify the INPUT into a single field: {task}.
Return STRICT JSON with only one key: "{task}", and the value MUST be one of {enums}.
Definition: {guidance}
Return ONLY the JSON object."""

    return {"enums": enums, "rag_system_prompt": rag_prompt, "no_rag_system_prompt": no_rag_prompt}

def build_messages_for(sample: Dict[str, Any], system_prompt: str, retriever: RAGRetriever = None, task: str = None) -> str:
    query_text = sample.get("data", "")
    if USE_RAG and retriever and task:
        sims = retriever.retrieve_balanced_samples(query_text, task, k=NUM_EXAMPLES) if USE_BALANCED_RAG else retriever.retrieve_similar_samples(query_text, k=NUM_EXAMPLES)
        blocks = [f"Input: {s.get('data', '')}\nOutput: {{\"{task}\": \"{s.get(task, 'N/A')}\"}}" for s in sims]
        sys_prompt = system_prompt.replace("{examples}", "\n\n".join(blocks))
        return f"{sys_prompt}\n\nUser Input: {query_text}\n\nClassification:"
    return f"{system_prompt}\n\nUser Input: {query_text}\n\nClassification:"

async def call_ollama_async(prompt: str, rate_limiter: AsyncRateLimiter) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            await rate_limiter.acquire()
            payload = {"model": MODEL, "prompt": prompt, "format": "json", "stream": False, "options": {"temperature": TEMPERATURE}}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)) as session:
                async with session.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload) as resp:
                    if resp.status != 200: raise Exception(f"API 錯誤: {resp.status}")
                    res = await resp.json()
                    return json.loads(res.get("response", "{}"))
        except Exception:
            if attempt == MAX_RETRIES: return {}
            await asyncio.sleep(1 * attempt)
    return {}

def load_json_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data

async def run_task_async(task: str, samples: List[Dict[str, Any]], out_dir: str, retriever: RAGRetriever = None):
    cfg = get_task_config(task)
    rate_limiter = AsyncRateLimiter(REQUESTS_PER_MINUTE)
    out_path = pathlib.Path(out_dir) / f"vpesg_{task}_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async def process(s, i):
        async with semaphore:
            input_text = s.get("data", "")
            sys_prompt = cfg["rag_system_prompt"] if (USE_RAG and retriever) else cfg["no_rag_system_prompt"]
            prompt = build_messages_for(s, sys_prompt, retriever, task)
            raw_pred = await call_ollama_async(prompt, rate_limiter)
            
            # --- 修改處：輸出加入 data 欄位 ---
            return {
                "i_id": s.get("i_id"),
                "data": input_text,  # 原始文本
                "pred": raw_pred     # 模型的 JSON 預測結果
            }

    results = []
    total_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx, i in enumerate(range(0, len(samples), BATCH_SIZE), 1):
        print(f"[{task}] 正在處理第 {batch_idx} / {total_batches} 批次 (索引: {i} 至 {min(i + BATCH_SIZE, len(samples))})")
        batch = [process(s, idx) for idx, s in enumerate(samples[i:i+BATCH_SIZE], i)]
        results.extend(await tqdm.gather(*batch, desc=f"{task} 進度控制"))

    with open(out_path, "w", encoding="utf-8") as wf:
        for r in results: 
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")

async def main():
    test_samples = load_json_samples(TEST_PATH)
    print(f"載入測試集完成，共 {len(test_samples)} 筆資料。")

    retriever = None
    if USE_RAG:
        retriever = RAGRetriever(EMBEDDING_MODEL)
        train_samples = load_json_samples(TRAIN_PATH)
        print(f"載入訓練集完成，正在建立 RAG 索引...")
        retriever.build_index(train_samples)

    tasks = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]
    for t in tasks:
        await run_task_async(t, test_samples, OUT_DIR, retriever)

if __name__ == "__main__":
    asyncio.run(main())