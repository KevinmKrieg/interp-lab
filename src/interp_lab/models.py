from __future__ import annotations

from interp_lab.hf_model import HuggingFaceCausalLM
from interp_lab.toy_model import ToyHookedTransformer


def make_model(config: dict):
    backend = config["model"].get("backend", "toy")
    if backend == "toy":
        model_config = config["model"]
        return ToyHookedTransformer(
            n_layers=int(model_config["n_layers"]),
            d_model=int(model_config["d_model"]),
            seed=int(config["seed"]),
        )
    if backend == "hf":
        return HuggingFaceCausalLM(config)
    raise ValueError(f"Unknown model backend: {backend}")
