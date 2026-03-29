import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

# vpesg4k verification_timeline 值的對應表
TIMELINE_MAP = {
    "already":              "Already",
    "within_2_years":       "Less than 2 years",
    "between_2_and_5_years": "2 to 5 years",
    "longer_than_5_years":  "More than 5 years",
    "N/A":                  "N/A",
}

# ── 讀資料 ──────────────────────────────────────────────
def load_counts(path, normalize=False):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # 若需要正規化 timeline 值（vpesg4k 用底線命名）
    if normalize:
        for d in data:
            v = d.get("verification_timeline", "N/A")
            d["verification_timeline"] = TIMELINE_MAP.get(v, v)

    order = {
        "promise_status":        ["Yes", "No"],
        "evidence_status":       ["Yes", "No", "N/A"],
        "evidence_quality":      ["Clear", "Not Clear", "Misleading", "N/A"],
        "verification_timeline": ["Already", "Less than 2 years",
                                  "2 to 5 years", "More than 5 years", "N/A"],
    }

    labels, counts = [], []
    for field, cats in order.items():
        c = Counter(d.get(field) for d in data)
        for cat in cats:
            # 若該類別count為0且不在資料中則跳過（保持彈性）
            labels.append(f"{field.replace('_', ' ').title()}\n{cat}")
            counts.append(c.get(cat, 0))
    return labels, counts, len(data)


# ── 顏色對應（同原圖：藍/橘/綠/黃） ─────────────────────
GROUP_COLORS = {
    "Promise Status":        "#4472C4",   # 藍
    "Evidence Status":       "#ED7D31",   # 橘
    "Evidence Quality":      "#70AD47",   # 綠
    "Verification Timeline": "#FFC000",   # 黃
}

GROUP_SIZES = [2, 3, 4, 5]   # 每組欄位數（evidence_status 含 N/A 共3項）

def make_colors(labels):
    colors = []
    group_names = list(GROUP_COLORS.keys())
    gi = 0
    used = 0
    for i in range(len(labels)):
        if used >= GROUP_SIZES[gi]:
            gi += 1
            used = 0
        colors.append(GROUP_COLORS[group_names[gi]])
        used += 1
    return colors


# ── 單張圖 ───────────────────────────────────────────────
def plot_chart(ax, labels, counts, title, total):
    colors = make_colors(labels)
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, width=0.6, zorder=2)

    # 數值標在柱頂
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(val), ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [lbl.split("\n")[1] + "\n(" + lbl.split("\n")[0] + ")" for lbl in labels],
        fontsize=7.5, rotation=30, ha="right"
    )
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)



# ── 主程式 ───────────────────────────────────────────────
ch_labels, ch_counts, ch_total = load_counts("dataset/Chinese_test.json")
en_labels, en_counts, en_total = load_counts("dataset/English_test.json")
vp_labels, vp_counts, vp_total = load_counts("dataset/vpesg4k_test_2000.json", normalize=True)

for labels, counts, total, title, out_path in [
    (ch_labels, ch_counts, ch_total, "ML_Promise Test Set (Chinese)", "label_distribution_chinese.png"),
    (en_labels, en_counts, en_total, "ML_Promise Test Set (English)", "label_distribution_english.png"),
    (vp_labels, vp_counts, vp_total, "VPesg4K Test Set",             "label_distribution_vpesg4k.png"),
]:
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_chart(ax, labels, counts, title, total)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()
