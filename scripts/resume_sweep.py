#!/usr/bin/env python
"""Resume the sweep from d=20, n_samples=1000, rep=3.

Reads config.json from the output directory, advances the master RNG to the
correct state, and appends rows to the existing metrics.csv and logs.txt
without touching anything that is already there.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from mcd import DAGConfig, ExpFamilyConfig, dtop, sample_multienv_scm
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


def main() -> None:
    outdir = Path("outputs/sweep")
    config = json.loads((outdir / "config.json").read_text())

    d_list = [int(x) for x in config["d_list"].split(",")]
    n_samples_list = [int(x) for x in config["n_samples"].split(",")]
    n_reps = config["n_reps"]
    seed = config["seed"]
    algorithm_names = parse_algorithm_list(config["algorithms"])
    krr_reg_list = [float(x) for x in config["krr_reg"].split(",")]
    eta_stein_list = [float(x) for x in config["eta_stein"].split(",")]
    densities = [float(x) for x in str(config["density"]).split(",")]

    exp_family_cfg = ExpFamilyConfig(
        family=config["exp_family"],
        shift_all_eta=config["shift_all_eta"],
        min_env_diff=config["min_env_diff"],
    )

    # --- target: d=20, n_samples=1000, reps 3..n_reps-1 ---
    TARGET_D = 20
    TARGET_N = 1000
    START_REP = 3

    # Advance master RNG past all completed draws.
    # Loop order mirrors run_experiment.py exactly:
    #   for d in d_list: for n_samples in n_samples_list: for rep in range(n_reps)
    rng_master = np.random.default_rng(seed)
    for d in d_list:
        for n_samples in n_samples_list:
            for rep in range(n_reps):
                if d == TARGET_D and n_samples == TARGET_N and rep == START_REP:
                    # Stop here — next call will yield the rep=3 seed
                    break
                rng_master.integers(0, 2**31 - 1)
            else:
                continue
            break
        else:
            continue
        break

    # Append to existing logs and metrics — never overwrite.
    log_file = open(outdir / "logs.txt", "a")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    csv_path = outdir / "metrics.csv"
    density = densities[-1] if len(densities) > 1 else densities[0]  # d=20 is last

    for rep in range(START_REP, n_reps):
        data_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(data_seed)
        data = sample_multienv_scm(
            k=config["n_envs"],
            n=TARGET_N,
            dag_config=DAGConfig(d=TARGET_D, density=density, graph_type="erdos_renyi"),
            mechanism=config["mechanism"],
            rng=rng,
            exp_family_cfg=exp_family_cfg,
        )

        print(f"d={TARGET_D}, n_samples={TARGET_N}, rep={rep}, edges={int(data.adjacency.sum())}")

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
                        denominator_eps=config["denominator_eps"],
                        score_train_frac=config["score_train_frac"],
                        aggregate_envs=config["aggregate_envs"],
                    )
                    t0 = time.perf_counter()
                    model = factory(alg_seed).fit(data.X)
                    elapsed_s = time.perf_counter() - t0
                    row = {
                        "algorithm": algorithm_name,
                        "rep": rep,
                        "data_seed": data_seed,
                        "algorithm_seed": alg_seed,
                        "d": TARGET_D,
                        "density": density,
                        "n_samples": TARGET_N,
                        "n_envs": config["n_envs"],
                        "mechanism": config["mechanism"],
                        "exp_family": config["exp_family"],
                        "shift_all_eta": config["shift_all_eta"],
                        "min_env_diff": config["min_env_diff"],
                        "krr_reg": krr_reg if is_mcd else "",
                        "eta_stein": eta_stein if is_mcd else "",
                        "score_train_frac": config["score_train_frac"],
                        "aggregate_envs": config["aggregate_envs"],
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

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writerows(rep_rows)

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Resume complete. Rows appended to {csv_path}")


if __name__ == "__main__":
    main()
