from __future__ import annotations

from pathlib import Path

import torch

from interp_lab.data import make_copy_prompts, target_token_for_prompt
from interp_lab.models import make_model


def collect_activations(config: dict) -> Path:
    model = make_model(config)
    prompts = make_copy_prompts(config["data"]["n_examples"], config["seed"])
    layer = int(config["data"]["layer_for_sae"])
    activation_rows = []
    metadata = []

    for prompt_index, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        for position in range(tokens.shape[0]):
            activation_rows.append(cache[layer, position])
            if config["model"].get("backend", "toy") == "toy":
                target_token_id = int(target_token_for_prompt(prompt))
            else:
                target_token_id = int(tokens[min(2, tokens.shape[0] - 1)])
            metadata.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "position": position,
                    "token_id": int(tokens[position]),
                    "token_text": model.token_label(int(tokens[position])),
                    "target_token_id": target_token_id,
                    "target_token_text": model.token_label(target_token_id),
                }
            )

    output = Path(config["outputs"]["artifacts_dir"]) / "activation_cache.pt"
    torch.save(
        {
            "layer": layer,
            "activations": torch.stack(activation_rows),
            "metadata": metadata,
            "d_model": model.d_model,
        },
        output,
    )
    return output
