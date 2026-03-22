#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async RAG-enhanced English ESG classifier (Ollama)
- Target Model: gemma3:27b via Ollama
- Optimized for English dataset (using "data" field)
- Logic synced with your localized paths
"""

import os, json, time, pathlib, re, sys
import numpy as np
import asyncio
import aiohttp
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.asyncio import tqdm

# ================ Config ================
MODEL = "gemma3:27b"   # Ollama 模型名稱
TEMPERATURE = 0 
MAX_RETRIES = 3

# Ollama 設定
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 120

# Concurrency / Rate limiting
MAX_CONCURRENT_REQUESTS = 8
REQUESTS_PER_MINUTE = 300
BATCH_SIZE = 25

# 檔案路徑 (英文版資料路徑)
TEST_PATH = r"C:\Users\lubob\Desktop\master thesis\dataset\English_test.json"
TRAIN_PATH = r"C:\Users\lubob\Desktop\master thesis\dataset\PromiseEval_Trainset_English.json"
OUT_DIR = r"C:\Users\lubob\Desktop\master thesis\results\Gemma3_4b\ML-Promise\English\NORAG"

# RAG 設定
USE_RAG = False
USE_BALANCED_RAG = True
NUM_EXAMPLES = 6
CANDIDATE_POOL_SIZE = 20
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
# ========================================

load_dotenv()

# ================ 工具類別 ================
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
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.train_samples: List[Dict[str, Any]] = []
        self.embeddings = None
        self.index = None
        print(f"[RAG] Embedding model running on: {self.embedding_model.device}")

    def build_index(self, train_samples: List[Dict[str, Any]]):
        print(f"[RAG] Building index over {len(train_samples)} training samples...")
        self.train_samples = train_samples
        # 英文版資料內容在 "data" 欄位
        texts = [s.get("data", "") for s in train_samples]
        self.embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        print(f"[RAG] Index ready, dim={d}")

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
            by_label.setdefault(s.get(task, "N/A"), []).append(s)
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

# ============ 任務配置與內容處理 ============
def get_task_config(task: str) -> Dict[str, Any]:
    if task == "promise_status":
        enums, guidance = ["Yes", "No"], "- promise_status: 'Yes' if the text expresses a concrete or organizational commitment; otherwise 'No'."
    elif task == "evidence_status":
        enums, guidance = ["Yes", "No"], "- evidence_status: 'Yes' if there is verifiable evidence/action; otherwise 'No'."
    elif task == "evidence_quality":
        enums = ["Clear", "Not Clear", "Misleading", "N/A"]
        guidance = "- evidence_quality: 'Clear' (specific/measurable), 'Not Clear' (vague), 'Misleading' (contradictory), 'N/A' (none)."
    elif task == "verification_timeline":
        enums = ["Already", "Less than 2 years", "2 to 5 years", "More than 5 years", "N/A"]
        guidance = "- verification_timeline: 'Already', 'Less than 2 years', '2 to 5 years', 'More than 5 years', 'N/A'."
    else:
        raise ValueError(f"Unsupported TASK: {task}")

    rag_system_prompt = f"""You are an ESG compliance inspector.
Classify the INPUT into a single field: {task}.
Return STRICT JSON with only one key: "{task}", and the value MUST be one of {enums}.
Definitions: {guidance}
Retrieved examples:
{{examples}}
Return ONLY the JSON object: {{"{task}": "value"}}"""

    no_rag_system_prompt = f"""You are an ESG compliance inspector.
Classify the INPUT into a single field: {task}.
Return STRICT JSON with only one key: "{task}", and the value MUST be one of {enums}.
Definitions: {guidance}
Return ONLY the JSON object: {{"{task}": "value"}}"""

    return {"enums": enums, "rag_system_prompt": rag_system_prompt, "no_rag_system_prompt": no_rag_system_prompt}

def build_messages_for(sample: Dict[str, Any], system_prompt: str, retriever: RAGRetriever = None, task: str = None) -> str:
    query_text = sample.get("data", "")
    if USE_RAG and retriever and task:
        sims = retriever.retrieve_balanced_samples(query_text, task, k=NUM_EXAMPLES) if USE_BALANCED_RAG else retriever.retrieve_similar_samples(query_text, k=NUM_EXAMPLES)
        blocks = [f"Example {i}:\nInput: {s.get('data', '')}\nOutput: {{\"{task}\": \"{s.get(task, 'N/A')}\"}}" for i, s in enumerate(sims, 1)]
        sys_prompt = system_prompt.replace("{examples}", "\n\n".join(blocks))
        return f"{sys_prompt}\n\nUser Input: {query_text}\n\nClassification:"
    return f"{system_prompt}\n\nUser Input: {query_text}\n\nClassification:"

# ============ Ollama 與 解析 ============
async def call_ollama_async(prompt: str, rate_limiter: AsyncRateLimiter) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            await rate_limiter.acquire()
            payload = {"model": MODEL, "prompt": prompt, "format": "json", "stream": False, "options": {"temperature": TEMPERATURE}}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)) as session:
                async with session.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload) as resp:
                    if resp.status != 200: raise Exception(f"Ollama error: {resp.status}")
                    res = await resp.json()
                    return json.loads(res.get("response", "{}"))
        except Exception as e:
            if attempt == MAX_RETRIES: raise e
            await asyncio.sleep(1 * attempt)
    return {}

def _normalize_pred_single(task: str, enums: List[str], d: Dict[str, Any]) -> Dict[str, str]:
    val = str(d.get(task, "")).strip()
    if val in enums: return {task: val}
    # 簡單後處理
    mapping = {"yes": "Yes", "no": "No", "n/a": "N/A", "clear": "Clear", "not clear": "Not Clear"}
    norm = mapping.get(val.lower(), "N/A" if "quality" in task or "timeline" in task else "No")
    return {task: norm}

# ============ 處理邏輯 ============
async def process_single_sample(sample: Dict[str, Any], idx: int, task: str, cfg: Dict[str, Any], retriever: RAGRetriever, rate_limiter: AsyncRateLimiter) -> Dict[str, Any]:
    try:
        input_text = sample.get("data", "")
        if not input_text: return {"idx": idx, "pred": {task: "N/A"}}
        
        sys_prompt = cfg["rag_system_prompt"] if (USE_RAG and retriever) else cfg["no_rag_system_prompt"]
        prompt = build_messages_for(sample, sys_prompt, retriever, task)
        raw_pred = await call_ollama_async(prompt, rate_limiter)
        return {"idx": idx, "input": {"data": input_text}, "pred": _normalize_pred_single(task, cfg["enums"], raw_pred)}
    except Exception as e:
        return {"idx": idx, "error": str(e)}

async def run_one_task_async(task: str, samples: List[Dict[str, Any]], out_dir: str, retriever: RAGRetriever = None):
    print(f"\n[TASK] {task} with Ollama ({MODEL})...")
    cfg = get_task_config(task)
    rate_limiter = AsyncRateLimiter(REQUESTS_PER_MINUTE)
    out_path = pathlib.Path(out_dir) / f"english_ollama_{task}_predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async def wrap(s, i):
        async with semaphore: return await process_single_sample(s, i, task, cfg, retriever, rate_limiter)

    results = []
    for i in range(0, len(samples), BATCH_SIZE):
        batch = [wrap(s, idx) for idx, s in enumerate(samples[i:i+BATCH_SIZE], i)]
        results.extend(await tqdm.gather(*batch, desc=f"{task} Batch {i//BATCH_SIZE + 1}"))

    with open(out_path, "w", encoding="utf-8") as wf:
        for r in results: wf.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{task}] DONE -> {out_path}")

def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
        return data if isinstance(data, list) else data.get("data", [])

# ============ Main ============
async def main():
    global USE_RAG
    print(f"🚀 Starting English ESG Prediction with {MODEL} (Ollama)")
    
    test_samples = load_samples(TEST_PATH)
    print(f"[INFO] Loaded {len(test_samples)} samples from English Test Set")

    retriever = None
    if USE_RAG:
        retriever = RAGRetriever(EMBEDDING_MODEL)
        if pathlib.Path(TRAIN_PATH).exists():
            train_samples = load_samples(TRAIN_PATH)
            print(f"[INFO] Loaded {len(train_samples)} samples from English Train Set")
            retriever.build_index(train_samples)
        else:
            print("[WARN] Train set not found, RAG disabled.")
            USE_RAG = False

    tasks = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]
    start = time.time()
    for t in tasks:
        await run_one_task_async(t, test_samples, OUT_DIR, retriever)
    
    print(f"\n🎉 ALL DONE. Total time: {(time.time()-start)/60:.1f} min")

if __name__ == "__main__":
    asyncio.run(main())