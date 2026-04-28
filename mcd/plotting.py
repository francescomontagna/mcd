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


def _format_hyperparam(krr: float, eta: float) -> str:
    return f"krr={krr:g}, η={eta:g}"


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
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "dtop_bars.png", dpi=180)
    plt.close(fig)


def plot_dtop_hyperparam_grid(results_csv: str, outdir: str) -> None:
    """One bar chart per (n_samples, mcd-variant).

    For each chart, the x-axis is the number of nodes d, and within each d
    there is one bar per krr_reg combination. Bar height is mean Dtop, error
    bars are 95% CIs.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    # key: (n_samples, algorithm, d, krr_reg, eta_stein) -> list of dtop values
    grouped: Dict[Tuple[int, str, int, float, float], List[float]] = defaultdict(list)
    n_samples_set: set[int] = set()
    d_set: set[int] = set()
    hyperparam_set: set[Tuple[float, float]] = set()
    algs_present: set[str] = set()

    for row in rows:
        algorithm = row["algorithm"]
        if algorithm not in MCD_VARIANTS_FOR_PLOT:
            continue
        if not row.get("krr_reg") or not row.get("eta_stein"):
            continue
        n_samples = int(row["n_samples"])
        d = int(row["d"])
        krr = float(row["krr_reg"])
        eta = float(row["eta_stein"])
        grouped[(n_samples, algorithm, d, krr, eta)].append(float(row["dtop"]))
        n_samples_set.add(n_samples)
        d_set.add(d)
        hyperparam_set.add((krr, eta))
        algs_present.add(algorithm)

    if not grouped:
        return

    n_samples_values = sorted(n_samples_set)
    d_values = sorted(d_set)
    # Sort hyperparameter combinations: first by krr_reg, then by eta_stein.
    hyperparam_combos = sorted(hyperparam_set)
    algs_to_plot = [a for a in MCD_VARIANTS_FOR_PLOT if a in algs_present]

    n_combos = len(hyperparam_combos)
    width = 0.8 / max(n_combos, 1)
    xs = np.arange(len(d_values))

    for n_samples in n_samples_values:
        for algorithm in algs_to_plot:
            fig, ax = plt.subplots()
            for i, (krr, eta) in enumerate(hyperparam_combos):
                means: List[float] = []
                cis: List[float] = []
                for d in d_values:
                    vals = grouped.get((n_samples, algorithm, d, krr, eta), [])
                    means.append(mean(vals) if vals else 0.0)
                    cis.append(_ci95(vals) if vals else 0.0)
                offsets = xs + (i - (n_combos - 1) / 2) * width
                ax.bar(offsets, means, width=width, yerr=cis, capsize=4,
                       label=_format_hyperparam(krr, eta))

            ax.set_xticks(xs)
            ax.set_xticklabels([str(d) for d in d_values])
            ax.set_xlabel("d (number of nodes)")
            ax.set_ylabel("mean Dtop")
            ax.set_title(f"{algorithm} | n_samples={n_samples}")
            ax.set_ylim(bottom=0.0)
            ax.legend(title="hyperparameters")
            fig.tight_layout()
            fname = f"dtop_{algorithm}_n{n_samples}.png"
            fig.savefig(out / fname, dpi=180)
            plt.close(fig)
