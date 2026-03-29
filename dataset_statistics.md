# Dataset Statistics — Quick Reference

> 用於實驗數據分析，所有數據集的類別分布一覽。

---

## 1. ML_Promise Dataset

### 1.1 Training Set

| Label | Category | Chinese (n=410) | English (n=400) |
|-------|----------|----------------:|----------------:|
| **Promise Status** | Yes | 146 | 313 |
| | No | 264 | 87 |
| **Evidence Status** | Yes | 78 | 221 |
| | No | 332 | 179 |
| **Evidence Quality** | Clear | 50 | 132 |
| | Not Clear | 16 | 85 |
| | Misleading | 1 | 4 |
| | N/A | 343 | 179 |
| **Verification Timeline** | Already | 0 | 155 |
| | Less than 2 years | 55 | 36 |
| | 2 to 5 years | 7 | 74 |
| | More than 5 years | 29 | 47 |
| | N/A | 319 | 87 |

> 注意：Chinese train set 的 Evidence Quality 有 1 筆標為 `Potentially Misleading`（非標準標籤）。

---

### 1.2 Test Set

| Label | Category | Chinese (n=489) | English (n=400) |
|-------|----------|----------------:|----------------:|
| **Promise Status** | Yes | 237 | 273 |
| | No | 252 | 127 |
| **Evidence Status** | Yes | 148 | 206 |
| | No | 341 | 194 |
| **Evidence Quality** | Clear | 73 | 134 |
| | Not Clear | 46 | 71 |
| | Misleading | 0 | 1 |
| | N/A | 370 | 194 |
| **Verification Timeline** | Already | 0 | 143 |
| | Less than 2 years | 101 | 36 |
| | 2 to 5 years | 11 | 50 |
| | More than 5 years | 39 | 44 |
| | N/A | 338 | 127 |

> 注意：Chinese test set 沒有 `Already` 類別。

---

## 2. VPesg4K Dataset

| Label | Category | Train (n=2000) | Test (n=2000) |
|-------|----------|---------------:|--------------:|
| **Promise Status** | Yes | 1626 | 1626 |
| | No | 374 | 374 |
| **Evidence Status** | Yes | 1342 | 1338 |
| | No | 284 | 288 |
| | N/A | 374 | 374 |
| **Evidence Quality** | Clear | 1116 | 1108 |
| | Not Clear | 225 | 228 |
| | Misleading | 1 | 2 |
| | N/A | 658 | 662 |
| **Verification Timeline** | Already | 730 | 737 |
| | Less than 2 years | 41 | 37 |
| | 2 to 5 years | 492 | 489 |
| | More than 5 years | 363 | 363 |
| | N/A | 374 | 374 |

---

## 3. 資料集規模總覽

| Dataset | Split | Size |
|---------|-------|-----:|
| ML_Promise Chinese | Train | 410 |
| ML_Promise Chinese | Test | 489 |
| ML_Promise English | Train | 400 |
| ML_Promise English | Test | 400 |
| VPesg4K | Train | 2000 |
| VPesg4K | Test | 2000 |
| **Total** | | **5699** |

---

## 4. 資料集差異備注

| 項目 | ML_Promise | VPesg4K |
|------|-----------|---------|
| 語言 | 中文 / 英文 | 英文 |
| Evidence Status 含 N/A | 否 | 是（= Promise Status 為 No 的樣本）|
| Chinese 含 Already | 否 | — |
| 標籤格式 | 正常字串 | Timeline 原始為底線命名（已正規化）|
| Promise Yes 比例 | Chinese: 49% / English: 68% | 81% |
