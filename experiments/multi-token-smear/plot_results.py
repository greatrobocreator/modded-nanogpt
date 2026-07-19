"""Standard loss plots for the multi-token smear full runs.

Usage: python plot_results.py <logs_dir> <out_dir>
Expects smear-k{K}-full-r{R}.txt logs (fetched from the Modal volume).
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

KS = (1, 2, 3)
REPEATS = (1, 2, 3)
COLORS = {1: "C0", 2: "C1", 3: "C2"}


def parse_val_curve(log_path: Path) -> list[tuple[int, float]]:
    text = log_path.read_text()
    return [
        (int(m.group(1)), float(m.group(2)))
        for m in re.finditer(r"^step:(\d+)/\d+ val_loss:([\d.]+) train_time", text, re.M)
    ]


def main(logs_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = {
        (k, r): parse_val_curve(logs_dir / f"smear-k{k}-full-r{r}.txt")
        for k in KS
        for r in REPEATS
    }
    assert all(curves.values()), "missing or empty logs"

    sns.set_theme(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for (k, r), pts in curves.items():
        steps, losses = zip(*[p for p in pts if p[0] > 0])
        ax1.plot(steps, losses, color=COLORS[k], alpha=0.45, lw=1.2,
                 label=f"k={k}" if r == 1 else None)
        tail = [(s, v) for s, v in pts if s >= 930]
        ax2.plot(*zip(*tail), color=COLORS[k], alpha=0.45, lw=1.2, marker="o", ms=3.5,
                 label=f"k={k}" if r == 1 else None)
    ax1.set(xlabel="step", ylabel="val loss", title="Val loss, full runs (3 repeats per k, 1×H100)")
    ax2.set(xlabel="step", ylabel="val loss", title="Tail zoom (steps 930–1390)")
    ax2.axhline(3.28, color="gray", lw=1, ls="--", label="3.28 speedrun target")
    for ax in (ax1, ax2):
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves_full.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    finals = {k: [curves[(k, r)][-1][1] for r in REPEATS] for k in KS}
    for i, k in enumerate(KS):
        ax.scatter([i] * len(finals[k]), finals[k], color=COLORS[k], s=55, zorder=3, label=f"k={k}")
        mean = sum(finals[k]) / len(finals[k])
        sd = (sum((v - mean) ** 2 for v in finals[k]) / (len(finals[k]) - 1)) ** 0.5
        ax.errorbar([i], [mean], yerr=[sd], color=COLORS[k], fmt="_", ms=26, capsize=7,
                    lw=2, zorder=2)
        ax.annotate(f"{mean:.4f}", (i, mean), textcoords="offset points",
                    xytext=(14, -4), fontsize=10)
    ax.axhline(3.28, color="gray", lw=1, ls="--", label="3.28 speedrun target")
    ax.set(xticks=range(len(KS)), xticklabels=[f"k={k}" for k in KS],
           ylabel="final val loss (step 1390)", title="Final val loss, mean ± sd over 3 repeats")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "final_loss_full.png", dpi=150)
    print(f"wrote {out_dir}/loss_curves_full.png and final_loss_full.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
