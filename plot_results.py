import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.patches import Patch

# ─────────────────────────────────────────────
# Raw data
# ─────────────────────────────────────────────
TASKS = ['promise_status', 'evidence_status', 'evidence_quality', 'verification_timeline']
TASK_LABELS = ['Promise\nStatus', 'Evidence\nStatus', 'Evidence\nQuality', 'Verification\nTimeline']

data = {
    # order: [promise_status, evidence_status, evidence_quality, verification_timeline]
    ('GPT5.4',    'ML-Promise_ZH', 'NORAG'): [0.8604, 0.5537, 0.3855, 0.2626],
    ('GPT5.4',    'ML-Promise_ZH', 'RAG'):   [0.8539, 0.5529, 0.3608, 0.3352],
    ('GPT5.4',    'ML-Promise_EN', 'NORAG'): [0.7150, 0.7406, 0.2647, 0.2933],
    ('GPT5.4',    'ML-Promise_EN', 'RAG'):   [0.7698, 0.7532, 0.4044, 0.4431],
    ('GPT5.4',    'VeriPromise',   'NORAG'): [0.7012, 0.3827, 0.2506, 0.3693],
    ('GPT5.4',    'VeriPromise',   'RAG'):   [0.7190, 0.3841, 0.3495, 0.4012],

    ('Gemma3-27b','ML-Promise_ZH', 'NORAG'): [0.8969, 0.5884, 0.4370, 0.2134],
    ('Gemma3-27b','ML-Promise_ZH', 'RAG'):   [0.9652, 0.7753, 0.5507, 0.4286],
    ('Gemma3-27b','ML-Promise_EN', 'NORAG'): [0.7454, 0.7390, 0.2758, 0.3131],
    ('Gemma3-27b','ML-Promise_EN', 'RAG'):   [0.7304, 0.7629, 0.4181, 0.3675],
    ('Gemma3-27b','VeriPromise',   'NORAG'): [0.6062, 0.4024, 0.3517, 0.2621],
    ('Gemma3-27b','VeriPromise',   'RAG'):   [0.6486, 0.4707, 0.3543, 0.3809],

    ('Gemma3-4b', 'ML-Promise_ZH', 'NORAG'): [0.7249, 0.4127, 0.4053, 0.3475],
    ('Gemma3-4b', 'ML-Promise_ZH', 'RAG'):   [0.9713, 0.8036, 0.4110, 0.4814],
    ('Gemma3-4b', 'ML-Promise_EN', 'NORAG'): [0.5469, 0.6622, 0.2042, 0.1730],
    ('Gemma3-4b', 'ML-Promise_EN', 'RAG'):   [0.6438, 0.6892, 0.2630, 0.2990],
    ('Gemma3-4b', 'VeriPromise',   'NORAG'): [0.6013, 0.3647, 0.2495, 0.0690],
    ('Gemma3-4b', 'VeriPromise',   'RAG'):   [0.5077, 0.3935, 0.2749, 0.2516],

    ('RoBERTa',   'VeriPromise',   'NORAG'): [0.7936, 0.6535, 0.4410, 0.5528],
}

MODELS    = ['GPT5.4', 'Gemma3-27b', 'Gemma3-4b']
DATASETS  = ['ML-Promise_ZH', 'ML-Promise_EN', 'VeriPromise']
DS_LABELS = ['ML-Promise (Chinese)', 'ML-Promise (English)', 'VeriPromise-4K']
DS_SHORT  = ['ML-Promise_ZH', 'ML-Promise_EN', 'VeriPromise']

COLORS = {
    'GPT5.4':     '#2196F3',
    'Gemma3-27b': '#FF9800',
    'Gemma3-4b':  '#4CAF50',
    'RoBERTa':    '#9C27B0',
}

# Legend order: large to small model (top to bottom), RoBERTa on top when present
MODELS_LEGEND_ORDER = ['GPT5.4', 'Gemma3-27b', 'Gemma3-4b']

LEGEND_HANDLES_RQ1 = (
    [Patch(facecolor=COLORS[m], label=m) for m in MODELS_LEGEND_ORDER]
    + [Patch(facecolor='gray', alpha=0.95, edgecolor='black', label='RAG'),
       Patch(facecolor='gray', alpha=0.45, edgecolor='black', label='NORAG')]
)

LEGEND_HANDLES_RQ2 = [Patch(facecolor=COLORS[m], edgecolor='black', label=m) for m in MODELS_LEGEND_ORDER]

LEGEND_HANDLES_RQ3 = (
    [Patch(facecolor=COLORS[m], edgecolor='black', label=m) for m in MODELS_LEGEND_ORDER]
    + [Patch(facecolor='gray', alpha=0.92, edgecolor='black', label='RAG (solid)'),
       Patch(facecolor='gray', alpha=0.40, edgecolor='black', label='NORAG (pale)')]
)

LEGEND_HANDLES_RQ3_ROBERTA = (
    [mlines.Line2D([], [], color=COLORS['RoBERTa'], marker='D',
                   linestyle='--', markersize=7, label='RoBERTa (fine-tuned)')]
    + LEGEND_HANDLES_RQ3
)

def avg(vals): return sum(vals) / len(vals)

# ═══════════════════════════════════════════════
# RQ1 — one figure per dataset
# ═══════════════════════════════════════════════
for ds, ds_label, ds_short in zip(DATASETS, DS_LABELS, DS_SHORT):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(f'RQ1: RAG vs NORAG — Avg. Macro-F1\n{ds_label}',
                 fontsize=12, fontweight='bold')

    x = np.arange(len(MODELS))
    w = 0.32
    norag_avgs = [avg(data[(m, ds, 'NORAG')]) for m in MODELS]
    rag_avgs   = [avg(data[(m, ds, 'RAG')])   for m in MODELS]
    deltas     = [r - n for r, n in zip(rag_avgs, norag_avgs)]

    ax.bar(x - w/2, norag_avgs, w,
           color=[COLORS[m] for m in MODELS], alpha=0.45, edgecolor='black', linewidth=0.7)
    bars_r = ax.bar(x + w/2, rag_avgs, w,
                    color=[COLORS[m] for m in MODELS], alpha=0.95, edgecolor='black', linewidth=0.7)

    for bar, d in zip(bars_r, deltas):
        sign = '+' if d >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.008,
                f'{sign}{d:.3f}', ha='center', va='bottom', fontsize=9, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel('Avg. Macro-F1', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=LEGEND_HANDLES_RQ1, fontsize=9, loc='upper left')

    plt.tight_layout()
    fname = f'statistic/RQ1_{ds_short}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

# ═══════════════════════════════════════════════
# RQ2 — one figure per dataset
# ═══════════════════════════════════════════════
for ds, ds_label, ds_short in zip(DATASETS, DS_LABELS, DS_SHORT):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f'RQ2: RAG Improvement (ΔMacro-F1) per Subtask\n{ds_label}',
                 fontsize=12, fontweight='bold')

    x = np.arange(len(TASKS))
    w = 0.22
    offsets = [-w, 0, w]

    for i, model in enumerate(MODELS):
        norag  = data[(model, ds, 'NORAG')]
        rag    = data[(model, ds, 'RAG')]
        deltas = [r - n for r, n in zip(rag, norag)]
        bars = ax.bar(x + offsets[i], deltas, w,
                      label=model, color=COLORS[model],
                      edgecolor='black', linewidth=0.7, alpha=0.85)
        for bar, d in zip(bars, deltas):
            if abs(d) > 0.005:
                va   = 'bottom' if d >= 0 else 'top'
                ypos = bar.get_height() + 0.003 if d >= 0 else bar.get_height() - 0.003
                ax.text(bar.get_x() + bar.get_width()/2, ypos,
                        f'{d:+.2f}', ha='center', va=va, fontsize=8, color='#222222')

    ax.axhline(0, color='black', linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, fontsize=10)
    ax.set_ylabel('ΔMacro-F1 (RAG − NORAG)', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=LEGEND_HANDLES_RQ2, fontsize=9, loc='upper left')

    plt.tight_layout()
    fname = f'statistic/RQ2_{ds_short}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

# ═══════════════════════════════════════════════
# RQ3 — one figure per dataset
# ═══════════════════════════════════════════════
for ds, ds_label, ds_short in zip(DATASETS, DS_LABELS, DS_SHORT):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f'RQ3: Model Scale & RAG Effect\n{ds_label}',
                 fontsize=12, fontweight='bold')

    x        = np.arange(len(TASKS))
    n_models = len(MODELS)
    group_w  = 0.70
    bar_w    = group_w / (n_models * 2)

    for mi, model in enumerate(MODELS):
        norag = data[(model, ds, 'NORAG')]
        rag   = data[(model, ds, 'RAG')]
        base  = x - group_w/2 + mi * (group_w / n_models) + bar_w * 0.2

        ax.bar(base,         norag, bar_w, color=COLORS[model], alpha=0.40,
               edgecolor='black', linewidth=0.5)
        ax.bar(base + bar_w, rag,   bar_w, color=COLORS[model], alpha=0.92,
               edgecolor='black', linewidth=0.5)

        for ti, (n_val, r_val) in enumerate(zip(norag, rag)):
            if abs(r_val - n_val) > 0.02:
                mid_x = base[ti] + bar_w / 2
                ax.annotate('', xy=(mid_x + bar_w, r_val), xytext=(mid_x, n_val),
                            arrowprops=dict(arrowstyle='->', color=COLORS[model],
                                            lw=1.3, alpha=0.75))

    # RoBERTa baseline only on VeriPromise
    if ds == 'VeriPromise':
        roberta_vals = data[('RoBERTa', 'VeriPromise', 'NORAG')]
        ax.plot(x, roberta_vals, 'D--', color=COLORS['RoBERTa'],
                markersize=7, linewidth=1.5, label='RoBERTa (fine-tuned)')
        legend_handles = LEGEND_HANDLES_RQ3_ROBERTA
    else:
        legend_handles = LEGEND_HANDLES_RQ3

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Macro-F1', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles, fontsize=9, loc='upper right')

    plt.tight_layout()
    fname = f'statistic/RQ3_{ds_short}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

print("\nAll individual figures saved to statistic/")
