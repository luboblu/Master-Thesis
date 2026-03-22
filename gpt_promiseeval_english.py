#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async RAG-enhanced English ESG classifier (GPT-4 via OpenAI API)
- Optimized for English dataset (using "data" field)
- Label enums EXACTLY aligned to English_test.json
"""

import os, json, time, pathlib, re, sys
import numpy as np
import asyncio
from typing import Dict, Any, List, Tuple
from collections import Counter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.asyncio import tqdm

# OpenAI (async client)
from openai import AsyncOpenAI

# ================ Config ================
MODEL = "gpt-5.4"        
TEMPERATURE = 0         # 分類任務建議設為 0 以求穩定
MAX_RETRIES = 3

# Concurrency / rate
MAX_CONCURRENT_REQUESTS = 8
REQUESTS_PER_MINUTE = 300
BATCH_SIZE = 25

# Paths - 英文版資料路徑
TEST_PATH = "C:\\Users\\lubob\\Desktop\\master thesis\\dataset\\English_test.json"
TRAIN_PATH = "C:\\Users\\lubob\\Desktop\\master thesis\\dataset\\PromiseEval_Trainset_English.json"
OUT_DIR = "C:\\Users\\lubob\\Desktop\\master thesis\\results\\GPT5.4\\ML-Promise\\English\\NORAG"

# RAG
USE_RAG = False
USE_BALANCED_RAG = True
NUM_EXAMPLES = 6
CANDIDATE_POOL_SIZE = 20
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
# ========================================

load_dotenv()
client = AsyncOpenAI()  # reads OPENAI_API_KEY

# ================ Utils ================
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

def get_sample_content(sample: Dict[str, Any]) -> str:
    """
    針對英文版：內容主要存放在 "data" 欄位
    """
    content = sample.get("data", "").strip()
    
    # 過濾掉無效內容
    invalid_markers = ["N/A", "nan", "None", "", "null"]
    if content in invalid_markers:
        return ""
    return content

class RAGRetriever:
    def __init__(self, embedding_model_name: str):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        self.train_samples: List[Dict[str, Any]] = []
        self.embeddings = None
        self.index = None
        print(f"Embedding model is running on: {self.embedding_model.device}")

    def build_index(self, train_samples: List[Dict[str, Any]]):
        print(f"[RAG] Building index over {len(train_samples)} training samples...")
        self.train_samples = train_samples
        
        # 使用英文版 data 欄位
        texts = [get_sample_content(s) for s in train_samples]
        # 確保訓練集沒有空文本干擾檢索
        processed_texts = [t if t else "No content available" for t in texts]
        
        self.embeddings = self.embedding_model.encode(processed_texts, convert_to_numpy=True)
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        print(f"[RAG] Index ready, dim={d}")

    def retrieve_similar_samples(self, query_text: str, k: int = 6) -> List[Dict[str, Any]]:
        if self.index is None or not query_text:
            return []
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
        if len(candidates) <= k:
            return candidates
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

# ============ Task config & Prompts ============
def get_task_config(task: str) -> Dict[str, Any]:
    if task == "promise_status":
        enums = ["Yes", "No"]
        guidance = "- promise_status: 'Yes' if the text expresses a concrete or organizational commitment; otherwise 'No'."
    elif task == "evidence_status":
        enums = ["Yes", "No"]
        guidance = "- evidence_status: 'Yes' if there is verifiable evidence/action; otherwise 'No'."
    elif task == "evidence_quality":
        enums = ["Clear", "Not Clear", "Misleading", "N/A"]
        guidance = "- evidence_quality: 'Clear' (specific/auditable), 'Not Clear' (vague), 'Misleading', 'N/A' (no evidence)."
    elif task == "verification_timeline":
        enums = ["Already", "Less than 2 years", "2 to 5 years", "More than 5 years", "N/A"]
        guidance = "- verification_timeline: 'Already', 'Less than 2 years', '2 to 5 years', 'More than 5 years', or 'N/A'."
    else:
        raise ValueError(f"Unsupported TASK: {task}")

    rag_system_prompt = f"""You are an ESG compliance inspector.
Classify the INPUT into a single field: {task}.
Return STRICT JSON with only one key: "{task}", and the value MUST be one of {enums}.
Definitions:
{guidance}

Here are examples for reference:
{{examples}}

Classify this input:"""

    no_rag_system_prompt = f"""You are an ESG compliance inspector.
Classify the INPUT into a single field: {task}.
Return STRICT JSON with only one key: "{task}", and the value MUST be one of {enums}.
Definitions:
{guidance}

Classify this input:"""

    return {"enums": enums, "rag_system_prompt": rag_system_prompt, "no_rag_system_prompt": no_rag_system_prompt}

def build_messages_for(sample: Dict[str, Any], system_prompt: str, retriever: RAGRetriever = None, task: str = None) -> str:
    query_text = get_sample_content(sample)
    
    if USE_RAG and retriever is not None and task is not None and query_text:
        sims = retriever.retrieve_balanced_samples(query_text, task, k=NUM_EXAMPLES) if USE_BALANCED_RAG else retriever.retrieve_similar_samples(query_text, k=NUM_EXAMPLES)
        blocks = []
        for i, s in enumerate(sims, 1):
            ex_content = get_sample_content(s)
            label = s.get(task, "N/A")
            blocks.append(f"Example {i}:\nInput: {ex_content}\nOutput: {{\"{task}\": \"{label}\"}}")
        sys_prompt = system_prompt.replace("{examples}", "\n\n".join(blocks))
        return f"{sys_prompt}\n\nUser Input: {query_text}\n\nClassification:"
    else:
        return f"{system_prompt}\n\nUser Input: {query_text}\n\nClassification:"

# ============ Parsing & Normalization ============
def _extract_json(text: str) -> Dict[str, Any]:
    text = re.sub(r'```json\s*|```\s*$', '', text).strip()
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m: raise ValueError(f"No JSON found: {text[:100]}")
    return json.loads(m.group(0))

def _normalize_pred_single(task: str, enums: List[str], d: Dict[str, Any]) -> Dict[str, str]:
    val = str(d.get(task, "")).strip()
    # 基本映射略...
    norm = val if val in enums else "No" if task in ["promise_status", "evidence_status"] else "N/A"
    return {task: norm}

async def call_gpt_async(prompt: str, rate_limiter: AsyncRateLimiter) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            await rate_limiter.acquire()
            resp = await client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": "You are a precise JSON classifier."},
                          {"role": "user", "content": prompt}]
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt >= MAX_RETRIES: raise e
            await asyncio.sleep(1.0 * attempt)
    return {}

# ============ Processing ============
async def process_single_sample(sample: Dict[str, Any], idx: int, task: str,
                                cfg: Dict[str, Any], retriever: RAGRetriever,
                                rate_limiter: AsyncRateLimiter) -> Dict[str, Any]:
    try:
        input_text = get_sample_content(sample)
        if not input_text:
            return {"idx": idx, "input": {"text": ""}, "pred": {task: "No" if "status" in task else "N/A"}}

        system_prompt = cfg["rag_system_prompt"] if (USE_RAG and retriever is not None) else cfg["no_rag_system_prompt"]
        prompt = build_messages_for(sample, system_prompt, retriever, task)
        raw_pred = await call_gpt_async(prompt, rate_limiter)
        pred = _normalize_pred_single(task, cfg["enums"], raw_pred)
        return {"idx": idx, "input": {"text": input_text}, "pred": pred}
    except Exception as e:
        return {"idx": idx, "error": str(e)}

async def run_one_task_async(task: str, samples: List[Dict[str, Any]], out_dir: str, retriever: RAGRetriever = None):
    print(f"\n[TASK] {task} with {MODEL}...")
    cfg = get_task_config(task)
    rate_limiter = AsyncRateLimiter(REQUESTS_PER_MINUTE)
    method_name = f"{MODEL}_rag" if USE_RAG else f"{MODEL}_0shot"
    out_path = pathlib.Path(out_dir) / f"{method_name}_{task}_predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async def wrap(sample, idx):
        async with semaphore: return await process_single_sample(sample, idx, task, cfg, retriever, rate_limiter)

    results = []
    for i in range(0, len(samples), BATCH_SIZE):
        batch = [wrap(s, idx) for idx, s in enumerate(samples[i:i+BATCH_SIZE], i)]
        results.extend(await tqdm.gather(*batch, desc=f"{task} Batch {i//BATCH_SIZE + 1}"))

    with open(out_path, "w", encoding="utf-8") as wf:
        for r in results: wf.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{task}] DONE -> {out_path}")

def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    return obj if isinstance(obj, list) else obj.get("data", [])

async def main():
    global USE_RAG
    print(f"🚀 Starting English ESG Classification")
    
    test_samples = load_samples(TEST_PATH)
    print(f"[INFO] Loaded {len(test_samples)} test samples")

    retriever = None
    if USE_RAG and pathlib.Path(TRAIN_PATH).exists():
        retriever = RAGRetriever(EMBEDDING_MODEL)
        train_samples = load_samples(TRAIN_PATH)
        retriever.build_index(train_samples)
    
    tasks = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]
    for t in tasks:
        await run_one_task_async(t, test_samples, OUT_DIR, retriever)

if __name__ == "__main__":
    asyncio.run(main())