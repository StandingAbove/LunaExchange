from common.model import Config
from common.utils import read_config_yaml


def validate_production_config(config: Config) -> None:
    # Exactly one aggregator must be specified for MVP simplicity
    assert len(config.aggregators) == 1, "Exactly one aggregator must be specified"
    assert len(config.models) > 0, "At least one model must be specified"
    # portfolio_optimizers is optional; if present, allow one for MVP
    portfolio_optimizers = getattr(config, "optimizers", [])
    assert (
            isinstance(portfolio_optimizers, list) or portfolio_optimizers is None
    ), "optimizers must be a list if provided"
    if isinstance(portfolio_optimizers, list):
        assert len(portfolio_optimizers) <= 1, "At most one portfolio optimizer for MVP"

    # Ensure models and optimizers match between production and simulation configs
    prod_config = read_config_yaml("production/paper/config.yaml")
    sim_config = read_config_yaml("production/simulations/config.yaml")

    # Compare models
    prod_models = set(prod_config.models)
    sim_models = set(sim_config.models)
    assert prod_models == sim_models, (
        f"Models mismatch between production and simulation configs. "
        f"Production: {sorted(prod_models)}, Simulation: {sorted(sim_models)}"
    )

    # Compare optimizers
    prod_optimizers = set(getattr(prod_config, "optimizers", []))
    sim_optimizers = set(getattr(sim_config, "optimizers", []))
    assert prod_optimizers == sim_optimizers, (
        f"Optimizers mismatch between production and simulation configs. "
        f"Production: {sorted(prod_optimizers)}, Simulation: {sorted(sim_optimizers)}"
    )
