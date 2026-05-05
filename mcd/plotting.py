from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple


MCD_VARIANTS_FOR_PLOT = ("mcd-variance", "mcd-hessian")


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
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "dtop_bars.png", dpi=180)
    plt.close(fig)


def plot_dtop_hyperparam_grid(
    results_csv: str, outdir: str, n_samples_filter: int | None = None
) -> None:
    """One bar chart per n_samples (or just one chart if n_samples_filter is set).

    x-axis is the number of nodes d; within each d there is one bar per
    algorithm. Bar height is mean Dtop across all reps and hyperparameter
    combinations; error bars are 95% CIs.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # key: (n_samples, algorithm, d) -> list of dtop values
    grouped: Dict[Tuple[int, str, int], List[float]] = defaultdict(list)
    n_samples_set: set[int] = set()
    d_set: set[int] = set()
    algs_present: set[str] = set()

    for row in rows:
        n_samples = int(row["n_samples"])
        if n_samples_filter is not None and n_samples != n_samples_filter:
            continue
        d = int(row["d"])
        algorithm = row["algorithm"]
        grouped[(n_samples, algorithm, d)].append(float(row["dtop"]))
        n_samples_set.add(n_samples)
        d_set.add(d)
        algs_present.add(algorithm)

    if not grouped:
        return

    n_samples_values = sorted(n_samples_set)
    d_values = sorted(d_set)
    algs_to_plot = sorted(algs_present)

    n_algs = len(algs_to_plot)
    width = 0.8 / max(n_algs, 1)
    xs = np.arange(len(d_values))

    for n_samples in n_samples_values:
        fig, ax = plt.subplots()
        for i, algorithm in enumerate(algs_to_plot):
            means: List[float] = []
            cis: List[float] = []
            for d in d_values:
                vals = grouped.get((n_samples, algorithm, d), [])
                means.append(mean(vals) if vals else 0.0)
                cis.append(_ci95(vals) if vals else 0.0)
            offsets = xs + (i - (n_algs - 1) / 2) * width
            ax.bar(offsets, means, width=width, yerr=cis, capsize=4, label=algorithm)

        ax.set_xticks(xs)
        ax.set_xticklabels([str(d) for d in d_values])
        ax.set_xlabel("d (number of nodes)")
        ax.set_ylabel("mean Dtop")
        ax.set_title(f"n_samples={n_samples}")
        ax.set_ylim(0.0, 1.0)
        ax.legend(title="algorithm")
        fig.tight_layout()
        fname = f"dtop_n{n_samples}.png"
        fig.savefig(out / fname, dpi=180)
        plt.close(fig)
