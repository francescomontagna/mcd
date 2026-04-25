from __future__ import annotations

import numpy as np

from mcd import DAGConfig, ExpFamilyConfig, MCD, MCDConfig, RandomOrderBaseline, RandomOrderConfig, dtop, sample_multienv_scm


def test_mcd_uses_all_eval_samples_without_caps() -> None:
    rng = np.random.default_rng(123)
    data = sample_multienv_scm(
        k=4,
        n=24,
        dag_config=DAGConfig(d=3, density=0.35),
        mechanism="mixed",
        rng=rng,
        exp_family_cfg=ExpFamilyConfig(family="gaussian"),
    )
    model = MCD(MCDConfig(sink_rule="variance", random_state=123)).fit(data.X)
    assert sorted(model.order_) == list(range(3))
    assert model.adjacency_.shape == data.adjacency.shape
    assert 0.0 <= dtop(data.adjacency, model.order_) <= 1.0
    assert model.env_split_.eval_env == 3


def test_random_baseline_runs() -> None:
    rng = np.random.default_rng(123)
    data = sample_multienv_scm(k=3, n=12, dag_config=DAGConfig(d=4, density=0.3), rng=rng)
    model = RandomOrderBaseline(RandomOrderConfig(random_state=7)).fit(data.X)
    assert sorted(model.order_) == list(range(4))
    assert 0.0 <= dtop(data.adjacency, model.order_) <= 1.0
