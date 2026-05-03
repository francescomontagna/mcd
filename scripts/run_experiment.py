#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from mcd import DAGConfig, ExpFamilyConfig, dtop, sample_multienv_scm
from mcd.plotting import plot_dtop_hyperparam_grid
from mcd.registry import MCD_VARIANTS, make_algorithm_factory, parse_algorithm_list

_FIELDNAMES = [
    "algorithm", "rep", "data_seed", "algorithm_seed", "d", "density",
    "n_samples", "n_envs", "mechanism", "exp_family", "shift_all_eta",
    "min_env_diff", "krr_reg", "eta_stein", "score_train_frac",
    "aggregate_envs", "dtop", "true_order", "estimated_order", "elapsed_s",
]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _parse_int_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic multi-environment causal-discovery experiments.")
    parser.add_argument("--outdir", type=str, default="outputs/quick")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--d-list", type=str, default="5",
                        help="Comma-separated list of node counts, e.g. 2,5,10,20.")
    parser.add_argument("--density", type=str, default="0.3",
                        help="Comma-separated list of densities (edge probabilities) for each d in d_list.")
    parser.add_argument("--n-samples", type=str, default="1000",
                        help="Comma-separated list of sample sizes per environment, e.g. 1000,2000,3000.")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--mechanism", type=str, default="mixed",
                        help="Structural mechanism for each node. Choices: "
                             "linear (linear additive noise), "
                             "anm/tanh (nonlinear additive noise, tanh), "
                             "pnl (post-nonlinear), "
                             "lsnm/location_scale (location-scale noise), "
                             "tanh_cubic (non-additive tanh+cubic), "
                             "nn/neural_net (non-additive neural network), "
                             "gp (non-additive Gaussian process), "
                             "mixed (randomly samples from all of the above per node).")
    parser.add_argument("--algorithms", type=str, default="mcd-variance,mcd-hessian",
                        help=f"Comma-separated list, choose from: {','.join(MCD_VARIANTS.keys())},random.")

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
    parser.add_argument("--eta-stein", type=str, default="1e-3",
                        help="Comma-separated list of Stein bandwidths (eta_G=eta_H), e.g. 5e-2,1e-2,1e-3.")
    parser.add_argument("--krr-reg", type=str, default="1e-3",
                        help="Comma-separated list of KRR regularization values, e.g. 5e-2,1e-2,1e-3.")
    parser.add_argument("--denominator-eps", type=float, default=1e-3)
    parser.add_argument("--score-train-frac", type=float, default=0.5)
    parser.add_argument("--aggregate-envs", choices=["sum", "mean"], default="sum")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_file = open(outdir / "logs.txt", "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    rng_master = np.random.default_rng(args.seed)

    algorithm_names = parse_algorithm_list(args.algorithms)
    krr_reg_list = _parse_float_list(args.krr_reg)
    eta_stein_list = _parse_float_list(args.eta_stein)
    n_samples_list = _parse_int_list(args.n_samples)
    d_list = _parse_int_list(args.d_list)

    densities = [float(d) for d in args.density.split(",")]
    assert len(densities) == 1 or len(densities) == len(d_list), (
        "Provide either a single density value or one per d in d_list."
    )

    exp_family_cfg = ExpFamilyConfig(
        family=args.exp_family,
        shift_all_eta=args.shift_all_eta,
        min_env_diff=args.min_env_diff,
    )

    run_config = vars(args).copy()
    (outdir / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True))

    csv_path = outdir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()

    rows = []
    total_reps = len(d_list) * len(n_samples_list) * args.n_reps
    pbar = tqdm(total=total_reps, desc="Sweep", file=sys.__stdout__)

    for n_samples in n_samples_list:
        for i, d in enumerate(d_list):
            density = densities[i] if len(densities) > 1 else densities[0]
            for rep in range(args.n_reps):
                data_seed = int(rng_master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(data_seed)
                data = sample_multienv_scm(
                    k=args.n_envs,
                    n=n_samples,
                    dag_config=DAGConfig(d=d, density=density, graph_type="erdos_renyi"),
                    mechanism=args.mechanism,
                    rng=rng,
                    exp_family_cfg=exp_family_cfg,
                )

                print(f"d={d}, n_samples={n_samples}, rep={rep}, edges={int(data.adjacency.sum())}")

                rep_rows = []
                for alg_offset, algorithm_name in enumerate(algorithm_names):
                    is_mcd = algorithm_name in MCD_VARIANTS
                    krr_values = krr_reg_list if is_mcd else [float("nan")]
                    eta_values = eta_stein_list if is_mcd else [float("nan")]
                    for krr_idx, krr_reg in enumerate(krr_values):
                        for eta_idx, eta_stein in enumerate(eta_values):
                            alg_seed = (
                                data_seed + 10_000 * (alg_offset + 1)
                                + 100 * krr_idx + eta_idx
                            )
                            factory = make_algorithm_factory(
                                algorithm_name,
                                eta_g=eta_stein if is_mcd else 0.0,
                                eta_h=eta_stein if is_mcd else 0.0,
                                krr_reg=krr_reg if is_mcd else 0.0,
                                denominator_eps=args.denominator_eps,
                                score_train_frac=args.score_train_frac,
                                aggregate_envs=args.aggregate_envs,
                            )
                            t0 = time.perf_counter()
                            model = factory(alg_seed).fit(data.X)
                            elapsed_s = time.perf_counter() - t0
                            row = {
                                "algorithm": algorithm_name,
                                "rep": rep,
                                "data_seed": data_seed,
                                "algorithm_seed": alg_seed,
                                "d": d,
                                "density": density,
                                "n_samples": n_samples,
                                "n_envs": args.n_envs,
                                "mechanism": args.mechanism,
                                "exp_family": args.exp_family,
                                "shift_all_eta": args.shift_all_eta,
                                "min_env_diff": args.min_env_diff,
                                "krr_reg": krr_reg if is_mcd else "",
                                "eta_stein": eta_stein if is_mcd else "",
                                "score_train_frac": args.score_train_frac,
                                "aggregate_envs": args.aggregate_envs,
                                "dtop": dtop(data.adjacency, model.order_),
                                "true_order": json.dumps([int(x) for x in data.order]),
                                "estimated_order": json.dumps([int(x) for x in model.order_]),
                                "elapsed_s": elapsed_s,
                            }

                            rep_rows.append(row)
                            if is_mcd:
                                print(
                                    f"  algorithm={algorithm_name}, krr_reg={krr_reg:g}, "
                                    f"eta_stein={eta_stein:g}, dtop={row['dtop']:.4f}"
                                )
                print('\n --- \n')

                rows.extend(rep_rows)
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=_FIELDNAMES).writerows(rep_rows)

                pbar.update(1)

            current_rows = [
                row for row in rows
                if row['d'] == d and row['n_samples'] == n_samples
            ]
            if current_rows:
                alg_names = sorted({row['algorithm'] for row in current_rows})
                print(f"Average dtop for d={d}, n_samples={n_samples}:")
                for alg in alg_names:
                    alg_dtops = [row['dtop'] for row in current_rows if row['algorithm'] == alg]
                    print(f"  {alg}: {float(np.mean(alg_dtops)):.4f}")
                print(" ------------ \n")

        if rows and not args.no_plots:
            plot_dtop_hyperparam_grid(str(csv_path), str(outdir))

    pbar.close()

    if rows and not args.no_plots:
        plot_dtop_hyperparam_grid(str(csv_path), str(outdir))
    print(f"Saved metrics to {csv_path}")

    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
