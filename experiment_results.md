# Experiment Results — Macro F1 Score Summary

> 所有數值為 Macro F1 Score，四捨五入至小數點第四位。各表依 **Avg（四任務平均）** 由高到低排序。

---

## 1. ML-Promise Chinese Test Set (n=489)

| Model | Setting | Promise Status | Evidence Status | Evidence Quality | Verification Timeline | **Avg** |
|-------|---------|---------------:|----------------:|-----------------:|----------------------:|--------:|
| Gemma3 27b | RAG | 0.9652 | 0.7753 | 0.5507 | 0.4286 | **0.6800** |
| Gemma3 4b | RAG | 0.9713 | 0.8036 | 0.4110 | 0.4814 | **0.6668** |
| ML-Promise Paper | RAG (baseline) | 0.540 | 0.503 | 0.628 | 0.469 | **0.5350** |
| Gemma3 27b | NORAG | 0.8969 | 0.5884 | 0.4370 | 0.2134 | **0.5339** |
| GPT-5.4 | RAG | 0.8539 | 0.5529 | 0.3608 | 0.3352 | **0.5257** |
| GPT-5.4 | NORAG | 0.8604 | 0.5537 | 0.3855 | 0.2626 | **0.5156** |
| Gemma3 4b | NORAG | 0.7249 | 0.4127 | 0.4053 | 0.3475 | **0.4726** |
| ML-Promise Paper | NORAG (baseline) | 0.521 | 0.163 | 0.569 | 0.317 | **0.3925** |

---

## 2. ML-Promise English Test Set (n=400)

| Model | Setting | Promise Status | Evidence Status | Evidence Quality | Verification Timeline | **Avg** |
|-------|---------|---------------:|----------------:|-----------------:|----------------------:|--------:|
| ML-Promise Paper | RAG (baseline) | 0.866 | 0.757 | 0.467 | 0.693 | **0.6958** |
| ML-Promise Paper | NORAG (baseline) | 0.842 | 0.680 | 0.411 | 0.636 | **0.6423** |
| GPT-5.4 | RAG | 0.7698 | 0.7532 | 0.4044 | 0.4431 | **0.5926** |
| Gemma3 27b | RAG | 0.7304 | 0.7629 | 0.4181 | 0.3675 | **0.5697** |
| Gemma3 27b | NORAG | 0.7454 | 0.7390 | 0.2758 | 0.3131 | **0.5183** |
| GPT-5.4 | NORAG | 0.7150 | 0.7406 | 0.2647 | 0.2933 | **0.5034** |
| Gemma3 4b | RAG | 0.6438 | 0.6892 | 0.2630 | 0.2990 | **0.4738** |
| Gemma3 4b | NORAG | 0.5469 | 0.6622 | 0.2042 | 0.1730 | **0.3966** |

---

## 3. VPesg4K Test Set (n=2000)

| Model | Setting | Promise Status | Evidence Status | Evidence Quality | Verification Timeline | **Avg** |
|-------|---------|---------------:|----------------:|-----------------:|----------------------:|--------:|
| Chinese RoBERTa | Fine-tuned **(baseline)** | 0.8177 | 0.6802 | 0.4416 | 0.5909 | **0.6326** |
| Chinese RoBERTa | Fine-tuned **(資料擴增)** | 0.7936 | 0.6535 | 0.4410 | 0.5528 | **0.6102** |
| GPT-5.4 | RAG | 0.7190 | 0.3841 | 0.3495 | 0.4012 | **0.4635** |
| Gemma3 27b | RAG | 0.6486 | 0.4707 | 0.3543 | 0.3809 | **0.4636** |
| GPT-5.4 | NORAG | 0.7012 | 0.3827 | 0.2506 | 0.3693 | **0.4260** |
| Gemma3 27b | NORAG | 0.6062 | 0.4024 | 0.3517 | 0.2621 | **0.4056** |
| Gemma3 4b | RAG | 0.5077 | 0.3935 | 0.2749 | 0.2516 | **0.3569** |
| Gemma3 4b | NORAG | 0.6013 | 0.3647 | 0.2495 | 0.0690 | **0.3211** |

---

## 4. RAG 效益（RAG − NORAG）

### ML-Promise Chinese

| Model | Promise Status | Evidence Status | Evidence Quality | Verification Timeline |
|-------|---------------:|----------------:|-----------------:|----------------------:|
| GPT-5.4 | -0.0065 | -0.0008 | -0.0247 | +0.0726 |
| Gemma3 27b | +0.0683 | +0.1869 | +0.1137 | +0.2152 |
| Gemma3 4b | +0.2464 | +0.3909 | +0.0057 | +0.1339 |

### ML-Promise English

| Model | Promise Status | Evidence Status | Evidence Quality | Verification Timeline |
|-------|---------------:|----------------:|-----------------:|----------------------:|
| GPT-5.4 | +0.0548 | +0.0126 | +0.1397 | +0.1498 |
| Gemma3 27b | -0.0150 | +0.0239 | +0.1423 | +0.0544 |
| Gemma3 4b | +0.0969 | +0.0270 | +0.0588 | +0.1260 |

### VPesg4K

| Model | Promise Status | Evidence Status | Evidence Quality | Verification Timeline |
|-------|---------------:|----------------:|-----------------:|----------------------:|
| GPT-5.4 | +0.0178 | +0.0014 | +0.0989 | +0.0319 |
| Gemma3 27b | +0.0424 | +0.0683 | +0.0026 | +0.1188 |
| Gemma3 4b | -0.0936 | +0.0288 | +0.0254 | +0.1826 |
