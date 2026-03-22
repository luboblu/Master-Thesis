#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async RAG-enhanced French ESG classifier (GPT-5 via OpenAI API)
- Prompts in English (generic)
- Label enums EXACTLY aligned to French_test.json
- Output normalization maps variants to test literals
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
MODEL = "gpt-5.4"       # OpenAI model name
TEMPERATURE = 1       # set 0 for stable classification
MAX_RETRIES = 3

# Concurrency / rate
MAX_CONCURRENT_REQUESTS = 8
REQUESTS_PER_MINUTE = 300
BATCH_SIZE = 25

# Paths (adjust as needed)
TEST_PATH = r"C:\Users\lubob\Desktop\master thesis\dataset\Chinese_test.json"
TRAIN_PATH = r"C:\Users\lubob\Desktop\master thesis\dataset\PromiseEval_Trainset_Chinese.json"
OUT_DIR = r"C:\Users\lubob\Desktop\master thesis\results\GPT5.4\ML-Promise\Chinese\NORAG"

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
        texts = [s.get("text", "") for s in train_samples]
        self.embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        print(f"[RAG] Index ready, dim={d}")

    def retrieve_similar_samples(self, query_text: str, k: int = 6) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
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
    """Enums EXACTLY match French_test.json options."""
    if task == "promise_status":
        enums = ["Yes", "No"]
        guidance = "- promise_status: 'Yes' if the text expresses a concrete or organizational commitment; otherwise 'No'."

    elif task == "evidence_status":
        enums = ["Yes", "No"]
        guidance = "- evidence_status: 'Yes' if there is verifiable evidence/action; otherwise 'No'."

    elif task == "evidence_quality":
        enums = ["Clear", "Not Clear", "Misleading", "N/A"]
        guidance = (
            "- evidence_quality:\n"
            "  * 'Clear': evidence is specific, measurable, auditable (dates, numbers, clear procedures).\n"
            "  * 'Not Clear': evidence exists but is vague or lacks details.\n"
            "  * 'Misleading': evidence contradicts targets or frames negative results as positive.\n"
            "  * 'N/A': no evidence provided."
        )

    elif task == "verification_timeline":
        enums = ["Already", "Less than 2 years", "2 to 5 years", "More than 5 years", "N/A"]
        guidance = (
            "- verification_timeline:\n"
            "  * 'Already': practice/plan already implemented.\n"
            "  * 'Less than 2 years': explicit timeline < 2 years.\n"
            "  * '2 to 5 years': explicit timeline between 2 and 5 years.\n"
            "  * 'More than 5 years': explicit timeline > 5 years.\n"
            "  * 'N/A': no timeline can be inferred."
        )
    else:
        raise ValueError(f"Unsupported TASK: {task}")

    rag_system_prompt = f"""You are an ESG compliance inspector.
Classify the INPUT into a single field: {task}.
Return STRICT JSON with only one key: "{task}", and the value MUST be one of {enums}.
No commentary, no explanation, no code fences.

Definitions:
{guidance}

Here are retrieved examples for reference:
{{examples}}

"Now classify this input:"
Return ONLY the JSON object like: {{"{task}": "value"}}
"""

    no_rag_system_prompt = f"""You are an ESG compliance inspector.
Classify the INPUT into a single field: {task}.
Return STRICT JSON with only one key: "{task}", and the value MUST be one of {enums}.
No commentary, no explanation, no code fences.

Definitions:
{guidance}

Now classify this input:
Return ONLY the JSON object like: {{"{task}": "value"}}
"""

    return {
        "enums": enums,
        "rag_system_prompt": rag_system_prompt,
        "no_rag_system_prompt": no_rag_system_prompt
    }


def build_examples_text(task: str, similar_samples: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, s in enumerate(similar_samples, 1):
        data = s.get("text", "")
        label = s.get(task, "N/A")
        sim = s.get("similarity_score", 0.0)
        blocks.append(
            f"Example {i} (similarity: {sim:.3f}):\nInput: {data}\nOutput: {{\"{task}\": \"{label}\"}}"
        )
    return "\n\n".join(blocks)

def get_sample_content(sample: Dict[str, Any]) -> str:
    """
    優先抓取 promise_string，如果為 N/A 或空值，則抓取 evidence_string
    """
    p_text = sample.get("promise_string", "")
    e_text = sample.get("evidence_string", "")
    
    # 邏輯：如果 promise_string 有內容且不是 "N/A"，就用它；否則用 evidence_string
    content = p_text if p_text and p_text != "N/A" else e_text
    
    # 如果兩者都是 "N/A" 或空，回傳空字串
    return content if content != "N/A" else ""
def build_messages_for(sample: Dict[str, Any], system_prompt: str,
                        retriever: "RAGRetriever" = None, task: str = None) -> str:
    
    # --- 修改這行：使用剛剛定義的函數來抓取內容 ---
    query_text = get_sample_content(sample) 
    
    if USE_RAG and retriever is not None and task is not None:
        if USE_BALANCED_RAG:
            sims = retriever.retrieve_balanced_samples(query_text, task, k=NUM_EXAMPLES, candidate_pool=CANDIDATE_POOL_SIZE)
        else:
            sims = retriever.retrieve_similar_samples(query_text, k=NUM_EXAMPLES)
        
        # --- 修改這裡：確保 RAG 抓出來的範例也是用正確的欄位 ---
        blocks = []
        for i, s in enumerate(sims, 1):
            ex_content = get_sample_content(s) # 範例也要抓對欄位
            label = s.get(task, "N/A")
            sim = s.get("similarity_score", 0.0)
            blocks.append(
                f"Example {i} (similarity: {sim:.3f}):\nInput: {ex_content}\nOutput: {{\"{task}\": \"{label}\"}}"
            )
        examples_text = "\n\n".join(blocks)
        
        sys_prompt = system_prompt.replace("{examples}", examples_text)
        return f"""{sys_prompt}\n\nUser Input: {query_text}\n\nClassification:"""
    else:
        return f"""{system_prompt}\n\nUser Input: {query_text}\n\nClassification:"""


# ============ Parsing & Normalization ============
def _extract_json(text: str) -> Dict[str, Any]:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = text.strip()
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(m.group(0))


def _normalize_pred_single(task: str, enums: List[str], d: Dict[str, Any]) -> Dict[str, str]:
    raw = d.get(task, "")
    val = str(raw).strip()
    low = val.lower()

    mapping = {
        # Binary
        "yes": "Yes", "no": "No",

        # CPEP
        "clear": "Clear",
        "not clear": "Not Clear", "not_clear": "Not Clear", "unclear": "Not Clear",
        "misleading": "Misleading",
        "n/a": "N/A", "na": "N/A", "none": "N/A",

        # TV -> test literals
        "already": "Already",
        "within 2 years": "Less than 2 years",
        "within_2_years": "Less than 2 years",
        "less than 2 years": "Less than 2 years",
        "less_than_2_years": "Less than 2 years",

        "2-5 years": "2 to 5 years",
        "2 to 5 years": "2 to 5 years",
        "between 2 and 5 years": "2 to 5 years",
        "between_2_and_5_years": "2 to 5 years",

        "longer than 5 years": "More than 5 years",
        "more than 5 years": "More than 5 years",
        "more_than_5_years": "More than 5 years",
        ">5 years": "More than 5 years",
        "over 5 years": "More than 5 years",
    }

    norm = mapping.get(low, val)

    if norm not in enums:
        if task == "verification_timeline":
            norm = "N/A"
        elif task in ("promise_status", "evidence_status"):
            norm = "No"
        elif task == "evidence_quality":
            norm = "N/A"

    return {task: norm}


# ============ GPT-5 Call ============
async def call_gpt_async(prompt: str, rate_limiter: AsyncRateLimiter) -> Dict[str, Any]:
    """
    Call OpenAI Chat Completions (GPT-5) and return a JSON object.
    We enforce JSON output via response_format and post-validate.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            await rate_limiter.acquire()
            resp = await client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise JSON-only classifier. "
                            "Always respond with a single valid JSON object, no extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content or ""
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return _extract_json(text)
        except Exception as e:
            print(f"[WARNING] Attempt {attempt} failed: {e}")
            if attempt >= MAX_RETRIES:
                raise e
            await asyncio.sleep(1.0 * attempt)
    return {}


# ============ Single/Batch Processing ============
async def process_single_sample(sample: Dict[str, Any], idx: int, task: str,
                                cfg: Dict[str, Any], retriever: RAGRetriever,
                                rate_limiter: AsyncRateLimiter) -> Dict[str, Any]:
    try:
        p_text = sample.get("promise_string", "")
        e_text = sample.get("evidence_string", "")
        

        input_text = ""
        if p_text and p_text != "N/A":
            input_text = p_text
        elif e_text and e_text != "N/A":
            input_text = e_text
        else:
            input_text = sample.get("text", "")
        sample_copy = sample.copy()
        sample_copy["text"] = input_text
        # -----------------------

        system_prompt = cfg["rag_system_prompt"] if (USE_RAG and retriever is not None) else cfg["no_rag_system_prompt"]
        

        prompt = build_messages_for(sample_copy, system_prompt, retriever, task)
        
        raw_pred = await call_gpt_async(prompt, rate_limiter)
        pred = _normalize_pred_single(task, cfg["enums"], raw_pred)
        
        return {
            "idx": idx, 
            "input": {"text": input_text}, 
            "pred": pred
        }
    except Exception as e:
        return {"idx": idx, "error": str(e)}


async def run_one_task_async(task: str, samples: List[Dict[str, Any]],
                             out_dir: str, retriever: RAGRetriever = None):
    # 1. 修改顯示日誌，反映真實使用的模型
    print(f"\n[TASK] {task} with {MODEL}...") 
    
    cfg = get_task_config(task)
    rate_limiter = AsyncRateLimiter(REQUESTS_PER_MINUTE)

    if USE_RAG and retriever is not None:
        method_name = f"{MODEL}_rag_balanced" if USE_BALANCED_RAG else f"{MODEL}_rag_standard"
    else:
        method_name = f"{MODEL}_0shot"

    # 這裡的檔名就會變成例如：gpt-5.4_rag_balanced_promise_status_predictions.jsonl
    out_path = pathlib.Path(out_dir) / f"{method_name}_{task}_predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async def wrap(sample, idx):
        async with semaphore:
            return await process_single_sample(sample, idx, task, cfg, retriever, rate_limiter)

    tasks_list = [wrap(sample, idx) for idx, sample in enumerate(samples)]
    print(f"[{task}] Processing {len(samples)} samples, concurrency={MAX_CONCURRENT_REQUESTS}...")

    results = []
    for i in range(0, len(tasks_list), BATCH_SIZE):
        batch = tasks_list[i:i+BATCH_SIZE]
        batch_res = await tqdm.gather(*batch, desc=f"{task} Batch {i//BATCH_SIZE + 1}")
        results.extend(batch_res)

    ok, fail = 0, 0
    with open(out_path, "w", encoding="utf-8") as wf:
        for r in results:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")
            if "error" in r: fail += 1
            else: ok += 1
    print(f"[{task}] DONE success={ok}, fail={fail} -> {out_path}")

# ============ I/O & Env ============
def load_samples(path: str) -> List[Dict[str, Any]]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if p.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line: out.append(json.loads(line))
        return out
    else:
        with open(p, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        if isinstance(obj, list): return obj
        if isinstance(obj, dict) and "data" in obj: return [obj]
        raise ValueError("Unsupported JSON structure. Expect list[dict] or jsonl.")


def explain_setup():
    print("\n" + "="*60)
    print("🚀 OpenAI GPT-5 setup")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Max concurrency: {MAX_CONCURRENT_REQUESTS}")
    print(f"Rate limit (client-side): {REQUESTS_PER_MINUTE} req/min")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*60 + "\n")


# ============ Self-check label sets ============
def print_label_sets_from_test(test_path: str):
    try:
        data = load_samples(test_path)
        keys = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]
        uniq = {k: set() for k in keys}
        for it in data:
            for k in keys:
                if k in it and it[k] is not None:
                    uniq[k].add(str(it[k]).strip())
        print("\n[CHECK] Unique label sets from test file:")
        for k in keys:
            print(f"  - {k}: {sorted(list(uniq[k]))}")
        print()
    except Exception as e:
        print(f"[CHECK] Failed to read test file: {e}")


# ============ Main ============
async def main():
    global USE_RAG

    explain_setup()

    # Self-check: print label sets from test
    print_label_sets_from_test(TEST_PATH)

    # Quick API key sanity check
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY".lower())):
        print("[ERROR] OPENAI_API_KEY not found in environment. Please set it or use a .env file.")
        return

    print(f"[INFO] MODEL={MODEL}")
    print(f"[INFO] USE_RAG={USE_RAG} | USE_BALANCED_RAG={USE_BALANCED_RAG if USE_RAG else '-'}")
    print(f"[INFO] TEST_PATH={TEST_PATH}")

    # 修改處 1: 讀取測試集並進入 "chinese_test" 層級
    raw_test = load_samples(TEST_PATH)
    if isinstance(raw_test, dict) and "chinese_test" in raw_test:
        test_samples = raw_test["chinese_test"]
    else:
        test_samples = raw_test
    print(f"[INFO] Loaded {len(test_samples)} test samples")

    retriever = None
    if USE_RAG:
        mode = "balanced" if USE_BALANCED_RAG else "standard"
        print(f"[INFO] Init RAG: {EMBEDDING_MODEL} ({mode})")
        retriever = RAGRetriever(EMBEDDING_MODEL)
        if pathlib.Path(TRAIN_PATH).exists():
            # 修改處 2: 讀取訓練集並進入 "chinese_train" 層級
            raw_train = load_samples(TRAIN_PATH)
            if isinstance(raw_train, dict) and "chinese_train" in raw_train:
                train_samples = raw_train["chinese_train"]
            else:
                train_samples = raw_train
                
            print(f"[INFO] Loaded {len(train_samples)} training samples for RAG")
            retriever.build_index(train_samples)
        else:
            print(f"[WARNING] TRAIN_PATH not found: {TRAIN_PATH}; disabling RAG")
            USE_RAG = False
            retriever = None

    tasks = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]

    start = time.time()
    for t in tasks:
        await run_one_task_async(t, test_samples, OUT_DIR, retriever)
    total = time.time() - start

    method = "balanced RAG" if (USE_RAG and USE_BALANCED_RAG) else ("standard RAG" if USE_RAG else "0-shot")
    print(f"\nDONE ({method}). Results saved to: {OUT_DIR}")
    print(f"Elapsed: {total/60:.1f} min")
    if total > 0:
        print(f"Throughput: {len(test_samples)*4/total:.1f} req/min")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[ABORTED] by user", file=sys.stderr)
        sys.exit(130)
