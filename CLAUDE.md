# CLAUDE.md

This repository implements synthetic experiments for multi-environment causal discovery.

Treat it as research code. Preserve mathematical clarity and reproducibility over abstraction.

## Core logic

MCD must use the following two-stage score-ratio pipeline:

1. Fit one Stein/KRR score model per training environment.
2. Evaluate all learned score models on the same evaluation dataset `X_eval`.
3. Form score-difference ratios pointwise on the evaluation points.
4. Identify sinks from the ratio criterion and peel recursively.

Do not implement the ratio by assuming paired raw observations across environments. The only required pairing is same-query evaluation of fitted score functions.

## Environment split

- `k = 3`: env 0 is base; envs 1 and 2 are auxiliary; env 0 is split into score-training and held-out evaluation subsets using `score_train_frac`.
- `k > 3`: env 0 is base; env `k-1` is the held-out evaluation environment; envs `1,...,k-2` are split alternately into auxiliary groups.

There are no training/evaluation sample caps. The user controls sample size through `--n-samples`; for `k=3`, `--score-train-frac` controls the base train/eval split.

## Repository conventions

- `X.shape == (num_environments, num_samples, num_nodes)`.
- `adjacency[child, parent] = 1` means `parent -> child`.
- `order_` is source-to-sink.
- Evaluate benchmark performance with `Dtop` only for now.
- Do not monitor SHD in `scripts/run_experiment.py`.
- Do not add pruning in MCD unless explicitly requested.

## Benchmarking

`scripts/run_experiment.py` accepts:

```bash
--algorithms mcd-variance,mcd-hessian,random
```

All listed algorithms must run on the same generated dataset for a given repetition. Add new algorithms through the registry in `mcd/registry.py`.

`--n-samples` and `--krr-reg` accept comma-separated lists; the script sweeps over their full cross product (along with `--d-list` and `--n-reps`) and writes every row to a single `metrics.csv`.

Current algorithms:

- `mcd-variance`: score-ratio sink peeling with the variance sink rule.
- `mcd-hessian`: score-ratio sink peeling with the hessian sink rule.
- `random`: random order baseline.

## Plotting

The plotting function should produce one bar chart per `n_samples`. Within each chart the x-axis is the number of nodes; each `d` group shows one bar per algorithm (all algorithms present in the CSV, averaged over hyperparameter combinations and repetitions). Bar height is mean `Dtop`; error bar is a 95% confidence interval.

## Before finishing changes

Run:

```bash
pytest -q
```

or at least a small smoke command such as:

```bash
python scripts/run_experiment.py --algorithms mcd-variance,mcd-hessian,random --n-reps 1 --d-list 3 --n-samples 20 --n-envs 4 --no-plots
```
