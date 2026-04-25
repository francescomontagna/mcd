#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np

from mcd import DAGConfig, ExpFamilyConfig, dtop, sample_multienv_scm
from mcd.plotting import plot_dtop_bars
from mcd.registry import make_algorithm_factories, parse_algorithm_list


def parse_d_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic multi-environment causal-discovery experiments.")
    parser.add_argument("--outdir", type=str, default="outputs/quick")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--d-list", type=str, default="5")
    parser.add_argument("--density", type=str, default="0.3", help="Comma-separated list of densities (edge probabilities) for each d in d_list.")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--mechanism", type=str, default="mixed")
    parser.add_argument("--algorithms", type=str, default="mcd,random", help="Comma-separated list, e.g. mcd,random.")

    # Exponential family source distribution options.
    parser.add_argument("--exp-family", type=str, default="gaussian",
                        help="Exponential family density type for source distributions (e.g. gaussian).")
    parser.add_argument("--shift-all-eta", action="store_true",
                        help="Shift all natural parameters eta; default shifts only the most meaningful one "
                             "(variance for gaussian).")
    parser.add_argument("--min-env-diff", type=float, default=1.0,
                        help="Minimum pairwise distance in standard-parameter space required across "
                             "at least 3 environments.")

    # MCD-specific options.
    parser.add_argument("--sink-rule", choices=["variance", "hessian"], default="hessian")
    parser.add_argument("--eta-g", type=float, default=1e-3)
    parser.add_argument("--eta-h", type=float, default=1e-3)
    parser.add_argument("--krr-reg", type=float, default=1e-3)
    parser.add_argument("--denominator-eps", type=float, default=1e-3)
    parser.add_argument("--score-train-frac", type=float, default=0.5)
    parser.add_argument("--aggregate-envs", choices=["sum", "mean"], default="sum")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(args.seed)

    algorithm_names = parse_algorithm_list(args.algorithms)
    factories = make_algorithm_factories(
        algorithm_names,
        sink_rule=args.sink_rule,
        eta_g=args.eta_g,
        eta_h=args.eta_h,
        krr_reg=args.krr_reg,
        denominator_eps=args.denominator_eps,
        score_train_frac=args.score_train_frac,
        aggregate_envs=args.aggregate_envs,
    )

    exp_family_cfg = ExpFamilyConfig(
        family=args.exp_family,
        shift_all_eta=args.shift_all_eta,
        min_env_diff=args.min_env_diff,
    )

    rows = []
    run_config = vars(args).copy()
    (outdir / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True))

    densities = [float(density) for density in args.density.split(",")]
    assert len(densities) == 1 or len(densities) == len(parse_d_list(args.d_list)), (
        "Provide either a single density value or one per d in d_list."
    )

    for i, d in enumerate(parse_d_list(args.d_list)):
        density = densities[i] if len(densities) > 1 else densities[0]
        for rep in range(args.n_reps):
            data_seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(data_seed)
            data = sample_multienv_scm(
                k=args.n_envs,
                n=args.n_samples,
                dag_config=DAGConfig(d=d, density=density, graph_type="erdos_renyi"),
                mechanism=args.mechanism,
                rng=rng,
                exp_family_cfg=exp_family_cfg,
            )

            print(f"Number of edges: {data.adjacency.sum()}")
            for alg_offset, (algorithm_name, factory) in enumerate(factories.items()):
                alg_seed = data_seed + 10_000 * (alg_offset + 1)
                model = factory(alg_seed).fit(data.X)
                row = {
                    "algorithm": algorithm_name,
                    "rep": rep,
                    "data_seed": data_seed,
                    "algorithm_seed": alg_seed,
                    "d": d,
                    "density": args.density,
                    "n_samples": args.n_samples,
                    "n_envs": args.n_envs,
                    "mechanism": args.mechanism,
                    "exp_family": args.exp_family,
                    "shift_all_eta": args.shift_all_eta,
                    "min_env_diff": args.min_env_diff,
                    "dtop": dtop(data.adjacency, model.order_),
                    "true_order": json.dumps([int(x) for x in data.order]),
                    "estimated_order": json.dumps([int(x) for x in model.order_]),
                }
                rows.append(row)
                print(f"d={d}, rep={rep}, algorithm={algorithm_name}, dtop={row['dtop']:.4f}")
            print('\n --- \n')

    csv_path = outdir / "metrics.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if rows and not args.no_plots:
        plot_dtop_bars(str(csv_path), str(outdir))
    print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
