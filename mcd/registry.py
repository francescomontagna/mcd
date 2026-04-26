from __future__ import annotations

from typing import Callable, Dict, Iterable, List

from .algorithm import MCD, MCDConfig
from .baselines import RandomOrderBaseline, RandomOrderConfig

AlgorithmFactory = Callable[[int], object]

# Algorithm names exposed via the registry. mcd-variance and mcd-hessian both
# instantiate MCD; only the sink_rule differs.
MCD_VARIANTS = {
    "mcd-variance": "variance",
    "mcd-hessian": "hessian",
}
AVAILABLE_ALGORITHMS = list(MCD_VARIANTS.keys()) + ["random"]


def parse_algorithm_list(text: str) -> List[str]:
    names = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not names:
        raise ValueError("At least one algorithm name is required.")
    return names


def make_algorithm_factory(
    name: str,
    *,
    eta_g: float,
    eta_h: float,
    krr_reg: float,
    denominator_eps: float,
    score_train_frac: float,
    aggregate_envs: str,
) -> AlgorithmFactory:
    key = name.strip().lower()
    if key in MCD_VARIANTS:
        sink_rule = MCD_VARIANTS[key]

        def factory(seed: int) -> MCD:
            return MCD(
                MCDConfig(
                    sink_rule=sink_rule,  # type: ignore[arg-type]
                    eta_G=eta_g,
                    eta_H=eta_h,
                    krr_reg=krr_reg,
                    denominator_eps=denominator_eps,
                    score_train_frac=score_train_frac,
                    aggregate_envs=aggregate_envs,  # type: ignore[arg-type]
                    random_state=seed,
                )
            )
        return factory
    if key == "random":
        def factory(seed: int) -> RandomOrderBaseline:
            return RandomOrderBaseline(RandomOrderConfig(random_state=seed))
        return factory
    raise ValueError(
        f"Unknown algorithm: {name}. Available algorithms: {', '.join(AVAILABLE_ALGORITHMS)}"
    )


def make_algorithm_factories(names: Iterable[str], **kwargs) -> Dict[str, AlgorithmFactory]:
    return {name: make_algorithm_factory(name, **kwargs) for name in names}


 
                              