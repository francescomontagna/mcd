from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


def _ci95(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return 1.96 * stdev(vals) / (len(vals) ** 0.5)


def plot_dtop_bars(results_csv: str, outdir: str) -> None:
    """One grouped-bar chart per run: x-axis = graph size d, one bar group per algorithm."""
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # grouped[d][alg] -> list of dtop values
    grouped: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["d"])][row["algorithm"]].append(float(row["dtop"]))

    d_values = sorted(grouped)
    algs = sorted({alg for d in grouped.values() for alg in d})
    n_algs = len(algs)
    width = 0.8 / n_algs
    xs = np.arange(len(d_values))

    fig, ax = plt.subplots()
    for i, alg in enumerate(algs):
        means = [mean(grouped[d][alg]) if alg in grouped[d] else 0.0 for d in d_values]
        cis = [_ci95(grouped[d][alg]) if alg in grouped[d] else 0.0 for d in d_values]
        offsets = xs + (i - (n_algs - 1) / 2) * width
        ax.bar(offsets, means, width=width, yerr=cis, capsize=4, label=alg)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in d_values])
    ax.set_xlabel("d (number of nodes)")
    ax.set_ylabel("mean Dtop")
    ax.set_title("Dtop by algorithm and graph size")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "dtop_bars.png", dpi=180)
    plt.close(fig)
