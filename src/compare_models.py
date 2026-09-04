import matplotlib.pyplot as plt
from pathlib import Path

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

baseline = [79.29, 79.00, 79.00, 79.00]
bert = [85.21, 85.75, 84.49, 85.12]

print("\nBASELINE vs BERT COMPARISON")
print("=" * 40)

for metric, b, t in zip(metrics, baseline, bert):
    print(
        f"{metric:<12} "
        f"Baseline: {b:>5.2f}%   "
        f"BERT: {t:>5.2f}%   "
        f"Improvement: {t - b:+.2f} pp"
    )

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(metrics))
width = 0.35

ax.bar(
    [i - width / 2 for i in x],
    baseline,
    width,
    label="Baseline"
)

ax.bar(
    [i + width / 2 for i in x],
    bert,
    width,
    label="BERT"
)

ax.set_ylabel("Score (%)")
ax.set_title("Baseline vs BERT Sentiment Classification")
ax.set_xticks(list(x))
ax.set_xticklabels(metrics)
ax.set_ylim(0, 100)
ax.legend()

for i, value in enumerate(baseline):
    ax.text(
        i - width / 2,
        value + 1,
        f"{value:.2f}",
        ha="center"
    )

for i, value in enumerate(bert):
    ax.text(
        i + width / 2,
        value + 1,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()

Path("reports").mkdir(exist_ok=True)

plt.savefig(
    "reports/model_comparison.png",
    dpi=300
)

plt.show()

print("\nComparison chart saved to:")
print("reports/model_comparison.png")