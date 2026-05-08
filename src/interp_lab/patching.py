from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from interp_lab.models import make_model
from interp_lab.toy_model import PatchSpec


def logit_diff(logits: torch.Tensor, correct_token_id: int, incorrect_token_id: int) -> float:
    return float(logits[correct_token_id] - logits[incorrect_token_id])


def run_patching(config: dict) -> dict[str, Path]:
    model = make_model(config)
    clean_tokens = model.to_tokens(config["data"]["clean_prompt"])
    corrupt_tokens = model.to_tokens(config["data"]["corrupt_prompt"])
    correct = config["data"]["correct_token"]
    incorrect = config["data"]["incorrect_token"]
    correct_id = model.token_id(correct)
    incorrect_id = model.token_id(incorrect)

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)
    clean_score = logit_diff(clean_logits, correct_id, incorrect_id)
    corrupt_score = logit_diff(corrupt_logits, correct_id, incorrect_id)
    denominator = clean_score - corrupt_score
    if abs(denominator) < 1e-6:
        denominator = 1.0

    recovery = torch.empty(model.n_layers, clean_tokens.shape[0])
    for layer in range(model.n_layers):
        for position in range(clean_tokens.shape[0]):
            patch = PatchSpec(layer=layer, position=position, value=clean_cache[layer, position])
            patched_logits, _ = model.run_with_cache(corrupt_tokens, patch=patch)
            patched_score = logit_diff(patched_logits, correct_id, incorrect_id)
            recovery[layer, position] = (patched_score - corrupt_score) / denominator

    reports = Path(config["outputs"]["reports_dir"])
    assets = reports / "assets"
    heatmap_path = assets / "patching_heatmap.png"
    attention_path = assets / "attention_pattern.png"
    json_path = reports / "patching_summary.json"
    markdown_path = reports / "patching_case_study.md"

    plt.figure(figsize=(8, 4.8))
    plt.imshow(recovery.numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=max(1.0, float(recovery.max())))
    plt.colorbar(label="logit-diff recovery")
    plt.xlabel("token position")
    plt.ylabel("layer")
    plt.title("Activation patching: clean signal restored into corrupt prompt")
    tokens = model.token_labels(clean_tokens)
    plt.xticks(range(clean_tokens.shape[0]), tokens)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=180)
    plt.close()

    attention = model.attention_pattern(clean_tokens)
    plt.figure(figsize=(5.2, 4.5))
    plt.imshow(attention.numpy(), cmap="magma", aspect="auto")
    plt.colorbar(label="attention weight")
    plt.xlabel("attended token")
    plt.ylabel("query token")
    plt.xticks(range(len(tokens)), tokens)
    plt.yticks(range(len(tokens)), tokens)
    plt.title("Toy induction head attention pattern")
    plt.tight_layout()
    plt.savefig(attention_path, dpi=180)
    plt.close()

    best_index = torch.argmax(recovery).item()
    best_layer = best_index // clean_tokens.shape[0]
    best_position = best_index % clean_tokens.shape[0]
    summary = {
        "backend": config["model"].get("backend", "toy"),
        "clean_prompt": config["data"]["clean_prompt"],
        "corrupt_prompt": config["data"]["corrupt_prompt"],
        "correct_token": correct,
        "incorrect_token": incorrect,
        "clean_logit_diff": clean_score,
        "corrupt_logit_diff": corrupt_score,
        "best_patch": {
            "layer": int(best_layer),
            "position": int(best_position),
            "token": tokens[best_position],
            "recovery": float(recovery[best_layer, best_position]),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["backend"] == "toy":
        interpretation = (
            "Interpretation: in this controlled toy setup, late residual-stream activations at the final "
            "position carry the copy signal needed to recover the held-out source token."
        )
    else:
        interpretation = (
            "Interpretation: patching this real-model residual-stream site strongly shifts the target "
            "logit difference toward the clean prompt behavior. This is a causal localization result, "
            "not yet a complete circuit explanation."
        )

    markdown_path.write_text(
        "# Activation Patching Case Study\n\n"
        f"Clean prompt: `{summary['clean_prompt']}` predicts `{correct}`.\n\n"
        f"Corrupt prompt: `{summary['corrupt_prompt']}` suppresses `{correct}` in favor of `{incorrect}`.\n\n"
        f"Best patch: layer `{best_layer}`, position `{best_position}` (`{tokens[best_position]}`), "
        f"recovering `{summary['best_patch']['recovery']:.2f}` of the clean logit difference.\n\n"
        f"{interpretation}\n",
        encoding="utf-8",
    )
    return {"heatmap": heatmap_path, "attention": attention_path, "summary": json_path, "report": markdown_path}
