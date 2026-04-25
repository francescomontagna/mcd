# MCD synthetic experiments

Synthetic experiments for multi-environment causal discovery.

The repository implements the following finite-sample logic explicitly:

1. For each training environment, fit a Stein/KRR score estimator
   `x -> grad_x log p^e(x)`.
2. Choose a common evaluation dataset `X_eval`.
3. Evaluate all learned score functions on the same `X_eval` points.
4. Form score-difference ratios pointwise on `X_eval`.
5. Use the ratio criterion to peel sinks recursively and recover a source-to-sink order.

This removes any need for paired raw observations across environments. What is paired is the evaluation of several learned score functions at the same query point `x`.

## Conventions

- `X.shape == (num_environments, num_samples, num_nodes)`.
- `adjacency[child, parent] = 1` means `parent -> child`.
- `order_` is source-to-sink.
- The main monitored metric is `Dtop`, the normalized topological-order violation rate.
- SHD and graph-pruning are intentionally not part of the current benchmark script.

## Install

```bash
pip install -e .
```

For tests:

```bash
pip install pytest
pytest -q
```

## Quick run

```bash
python scripts/run_experiment.py \
  --outdir outputs/debug \
  --algorithms mcd,random \
  --n-reps 2 \
  --d-list 5 \
  --n-samples 100 \
  --n-envs 4 \
  --sink-rule variance \
  --no-plots
```

A more serious run:

```bash
python scripts/run_experiment.py \
  --outdir outputs/hessian \
  --algorithms mcd,random \
  --seed 0 \
  --n-reps 10 \
  --d-list 5,10,20 \
  --density 0.3 \
  --n-samples 300 \
  --n-envs 4 \
  --mechanism mixed \
  --noise gaussian \
  --sink-rule hessian \
  --eta-g 1e-2 \
  --eta-h 1e-2 \
  --krr-reg 1e-4
```

## Environment logic

For `k = 4`, the default split is:

```text
env 0: base score model
env 1: auxiliary group 1 score model
env 2: auxiliary group 2 score model
env 3: common evaluation data X_eval
```

The ratio is computed by evaluating all score models on env 3 samples:

```text
score_0 = score_model_0(X_eval)
score_1 = score_model_1(X_eval)
score_2 = score_model_2(X_eval)
ratio_i = (score_0_i - score_1_i) / (score_0_i - score_2_i)
```

For `k = 3`, there is no fourth evaluation environment, so env 0 is split:

```text
env 0 train subset: fit base score model
env 0 held-out subset: common evaluation data X_eval
env 1: auxiliary group 1 score model
env 2: auxiliary group 2 score model
```

The split fraction is controlled by:

```bash
--score-train-frac 0.5
```

There are no `max_score_train` or `max_eval_samples` arguments. Training and evaluation sizes are controlled by `--n-samples` and, for `k=3`, by `--score-train-frac`.

## Benchmarking multiple algorithms

The script accepts a comma-separated algorithm list:

```bash
--algorithms mcd,random
```

Currently implemented:

- `mcd`: score-ratio sink peeling.
- `random`: random source-to-sink order baseline.

All algorithms are run on the same generated dataset for each repetition.

## Outputs

The experiment writes:

```text
outputs/.../config.json
outputs/.../metrics.csv
outputs/.../dtop_bars_d{d}.png
```

The plot is one bar chart per graph size. Each bar is one algorithm. The bar height is mean `Dtop`; the error bar is a 95% confidence interval over repetitions.
