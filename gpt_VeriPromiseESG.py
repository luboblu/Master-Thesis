#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高效能非同步 RAG ESG 分類器 (GPT5.4 穩定版本)
- 修復重點：環境變數強制去空格、啟動配置檢查、自動報錯追蹤
- 輸出內容：i_id, data (原始文本), pred (預測結果)
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

# ================ 載入環境變數與安全檢查 ================
load_dotenv()

# 強制去除可能存在的隱藏空格或換行符
MODEL = str(os.getenv("MODEL", "gpt-5.4")).strip()
API_KEY = str(os.getenv("OPENAI_API_KEY", "")).strip()
# 如果 .env 沒讀到，則給予預設官方網址作為備案
API_BASE_URL = str(os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")).strip()

# 啟動時的環境校驗
print("="*50)
print(f"[啟動檢查] 使用模型: {MODEL}")
print(f"[啟動檢查] API 網址: {API_BASE_URL}")
if not API_KEY:
    print("[嚴重錯誤] 找不到 API KEY，請檢查 .env 檔案位置是否與腳本相同！")
    sys.exit(1)
if not API_BASE_URL.startswith("http"):
    print("[嚴重錯誤] API 網址格式錯誤，必須以 http 開頭！")
    sys.exit(1)
print("="*50)

# ================ 執行參數設定 ================
TEMPERATURE = 0 
MAX_RETRIES = 5
OLLAMA_TIMEOUT = 120

# 效能調優：兼顧速度與穩定性
MAX_CONCURRENT_REQUESTS = 20  # 同時發送 20 個請求
REQUESTS_PER_MINUTE = 150
BATCH_SIZE = 40               # 每批次處理 40 筆樣本

# 檔案路徑
TEST_PATH = r"C:\Users\lubob\Desktop\master thesis\dataset\vpesg4k_test_2000.json"
TRAIN_PATH = r"C:\Users\lubob\Desktop\master thesis\dataset\vpesg4k_train_2000.json"
OUT_DIR = r"C:\Users\lubob\Desktop\master thesis\results\GPT5.4\VeriPromise\RAG"

# RAG 配置
USE_RAG = True
USE_BALANCED_RAG = True
NUM_EXAMPLES = 6
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
# ============================================

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
            print(f"\n[GPU 狀態] 偵測到 {torch.cuda.get_device_name(0)}，開啟 CUDA 加速。")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.train_samples = []
        self.index = None

    def build_index(self, train_samples: List[Dict[str, Any]]):
        self.train_samples = train_samples
        texts = [str(s.get("data", "")) for s in train_samples]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def retrieve_balanced_samples(self, query_text: str, task: str, k: int = 6) -> List[Dict[str, Any]]:
        if self.index is None: return []
        q = self.embedding_model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(q)
        _, idxs = self.index.search(q, 20)
        candidates = [self.train_samples[i] for i in idxs[0] if i < len(self.train_samples)]
        
        by_label = {}
        for s in candidates:
            label = str(s.get(task, "N/A"))
            by_label.setdefault(label, []).append(s)
        
        selected = []
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

async def call_gpt_async(system_prompt: str, user_content: str, rate_limiter: AsyncRateLimiter) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        "response_format": {"type": "json_object"},
        "temperature": TEMPERATURE
    }
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            await rate_limiter.acquire()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)) as session:
                # 確保傳入 session.post 的網址絕對是字串
                async with session.post(str(API_BASE_URL), headers=headers, json=payload) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(5 * attempt)
                        continue
                    if resp.status != 200:
                        err = await resp.text()
                        raise Exception(f"HTTP {resp.status}: {err}")
                    res = await resp.json()
                    return json.loads(res["choices"][0]["message"]["content"])
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"\n[API 錯誤] {str(e)}")
                return {}
            await asyncio.sleep(2 * attempt)
    return {}

async def run_task_async(task: str, samples: List[Dict[str, Any]], out_dir: str, retriever: RAGRetriever = None):
    configs = {
        "promise_status": (["Yes", "No"], "Classify commitment."),
        "evidence_status": (["Yes", "No"], "Classify evidence."),
        "evidence_quality": (["Clear", "Not Clear", "Misleading", "N/A"], "Assess quality."),
        "verification_timeline": (["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years", "N/A"], "Identify timeline.")
    }
    enums, guidance = configs[task]
    sys_prompt = f"You are an ESG inspector. Return STRICT JSON with key '{task}'. Value MUST be one of {enums}. Definition: {guidance}"
    
    rate_limiter = AsyncRateLimiter(REQUESTS_PER_MINUTE)
    out_path = pathlib.Path(out_dir) / f"vpesg_{task}_gpt_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async def process(s):
        async with semaphore:
            input_text = str(s.get("data", ""))
            user_content = f"Input: {input_text}"
            if USE_RAG and retriever:
                sims = retriever.retrieve_balanced_samples(input_text, task, k=NUM_EXAMPLES)
                examples = "\n\n".join([f"Input: {ex['data']}\nOutput: {json.dumps({task: ex[task]})}" for ex in sims])
                user_content = f"Examples:\n{examples}\n\n{user_content}"
            raw_pred = await call_gpt_async(sys_prompt, user_content, rate_limiter)
            return {"i_id": s.get("i_id"), "data": input_text, "pred": raw_pred}

    results = []
    total_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE
    for b_idx, i in enumerate(range(0, len(samples), BATCH_SIZE), 1):
        print(f"[{task}] 進度：{b_idx} / {total_batches} 批次 (已處理: {i} 筆)")
        batch = [process(s) for s in samples[i:i+BATCH_SIZE]]
        results.extend(await tqdm.gather(*batch, desc=f"{task} 並行執行中"))

    with open(out_path, "w", encoding="utf-8") as wf:
        for r in results: wf.write(json.dumps(r, ensure_ascii=False) + "\n")

async def main():
    try:
        with open(TEST_PATH, "r", encoding="utf-8") as f: test_samples = json.load(f)
    except Exception as e:
        print(f"[錯誤] 無法讀取測試檔案: {e}")
        return

    retriever = None
    if USE_RAG:
        retriever = RAGRetriever(EMBEDDING_MODEL)
        with open(TRAIN_PATH, "r", encoding="utf-8") as f: train_samples = json.load(f)
        print("[系統] 正在建立 RAG 向量索引...")
        retriever.build_index(train_samples)

    tasks = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]
    for t in tasks:
        await run_task_async(t, test_samples, OUT_DIR, retriever)
    print("\n🎉 分類任務全部完成！")

if __name__ == "__main__":
    asyncio.run(main())