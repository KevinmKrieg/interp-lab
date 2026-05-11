from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from interp_lab.data import make_induction_pairs, make_text_induction_pairs
from interp_lab.models import make_model
from interp_lab.toy_model import PatchSpec


def logit_diff(logits: torch.Tensor, correct_token_id: int, incorrect_token_id: int) -> float:
    return float(logits[correct_token_id] - logits[incorrect_token_id])


def _format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _save_heatmap(
    recovery: torch.Tensor,
    tokens: list[str],
    path: Path,
    title: str,
    *,
    vmin: float = 0.0,
    vmax: float | None = None,
) -> None:
    data = recovery.detach().cpu().numpy()
    masked = np.ma.masked_invalid(data)
    plt.figure(figsize=(8, 4.8))
    plt.imshow(masked, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax or max(1.0, float(np.nanmax(data))))
    plt.colorbar(label="logit-diff recovery")
    plt.xlabel("token position")
    plt.ylabel("layer")
    plt.title(title)
    plt.xticks(range(len(tokens)), tokens)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _mean_recovery(results: list[dict]) -> torch.Tensor:
    n_layers = results[0]["recovery"].shape[0]
    max_positions = max(item["recovery"].shape[1] for item in results)
    padded = torch.full((len(results), n_layers, max_positions), float("nan"))
    for index, item in enumerate(results):
        width = item["recovery"].shape[1]
        padded[index, :, :width] = item["recovery"]
    return torch.nanmean(padded, dim=0)


def _case_summary(item: dict) -> dict:
    recovery = item["recovery"]
    best_value = float(recovery[item["best_layer"], item["best_position"]])
    return {
        "clean_prompt": item["case"]["clean_prompt"],
        "corrupt_prompt": item["case"]["corrupt_prompt"],
        "correct_token": item["case"]["correct_token"],
        "incorrect_token": item["case"]["incorrect_token"],
        "clean_logit_diff": item["clean_score"],
        "corrupt_logit_diff": item["corrupt_score"],
        "best_layer": item["best_layer"],
        "best_position": item["best_position"],
        "best_token": item["tokens"][item["best_position"]],
        "best_recovery": best_value,
    }


def _patch_single(model, case: dict[str, str], min_clean_margin: float = 0.0) -> dict:
    clean_tokens = model.to_tokens(case["clean_prompt"])
    corrupt_tokens = model.to_tokens(case["corrupt_prompt"])
    correct = case["correct_token"]
    incorrect = case["incorrect_token"]
    correct_id = model.token_id(correct)
    incorrect_id = model.token_id(incorrect)

    if clean_tokens.shape != corrupt_tokens.shape:
        return {"valid": False, "case": case, "reason": "clean/corrupt token lengths differ"}

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupt_logits, _ = model.run_with_cache(corrupt_tokens)
    clean_score = logit_diff(clean_logits, correct_id, incorrect_id)
    corrupt_score = logit_diff(corrupt_logits, correct_id, incorrect_id)
    denominator = clean_score - corrupt_score
    if denominator <= min_clean_margin:
        return {
            "valid": False,
            "case": case,
            "clean_score": clean_score,
            "corrupt_score": corrupt_score,
            "reason": "clean logit margin does not exceed corrupt margin",
        }

    recovery = torch.empty(model.n_layers, clean_tokens.shape[0])
    for layer in range(model.n_layers):
        for position in range(clean_tokens.shape[0]):
            patch = PatchSpec(layer=layer, position=position, value=clean_cache[layer, position])
            patched_logits, _ = model.run_with_cache(corrupt_tokens, patch=patch)
            patched_score = logit_diff(patched_logits, correct_id, incorrect_id)
            recovery[layer, position] = (patched_score - corrupt_score) / denominator

    best_index = torch.argmax(recovery).item()
    best_layer = best_index // clean_tokens.shape[0]
    best_position = best_index % clean_tokens.shape[0]
    return {
        "case": case,
        "valid": True,
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "tokens": model.token_labels(clean_tokens),
        "recovery": recovery,
        "clean_score": clean_score,
        "corrupt_score": corrupt_score,
        "best_layer": int(best_layer),
        "best_position": int(best_position),
    }


def run_patching(config: dict) -> dict[str, Path]:
    model = make_model(config)
    backend = config["model"].get("backend", "toy")
    patching_dataset = config["data"].get("patching_dataset")
    if patching_dataset == "induction":
        cases = make_induction_pairs(int(config["data"].get("patching_examples", 8)), int(config["seed"]))
    elif patching_dataset == "text_induction":
        cases = make_text_induction_pairs()[: int(config["data"].get("patching_examples", 8))]
    else:
        cases = [
            {
                "clean_prompt": config["data"]["clean_prompt"],
                "corrupt_prompt": config["data"]["corrupt_prompt"],
                "correct_token": config["data"]["correct_token"],
                "incorrect_token": config["data"]["incorrect_token"],
            }
        ]

    min_clean_margin = float(config["data"].get("min_clean_margin", 0.0))
    results = [_patch_single(model, case, min_clean_margin) for case in cases]
    skipped_cases = [result for result in results if not result["valid"]]
    results = [result for result in results if result["valid"]]
    if not results:
        reasons = "; ".join(f"{item['case']['clean_prompt']}: {item['reason']}" for item in skipped_cases)
        raise ValueError(f"No valid patching cases. {reasons}")
    hero = max(results, key=lambda item: float(item["recovery"].max()))
    recovery = hero["recovery"]
    clean_tokens = hero["clean_tokens"]
    tokens = hero["tokens"]

    reports = Path(config["outputs"]["reports_dir"])
    assets = reports / "assets"
    heatmap_path = assets / "patching_heatmap.png"
    mean_heatmap_path = assets / "patching_heatmap_mean.png"
    attention_path = assets / "attention_pattern.png"
    json_path = reports / "patching_summary.json"
    markdown_path = reports / "patching_case_study.md"

    _save_heatmap(
        recovery,
        tokens,
        heatmap_path,
        "Hero case: clean signal restored into corrupt prompt",
    )

    mean_recovery = _mean_recovery(results)
    mean_tokens = [f"pos {index}" for index in range(mean_recovery.shape[1])]
    _save_heatmap(
        mean_recovery,
        mean_tokens,
        mean_heatmap_path,
        "Mean patching recovery across valid cases",
    )

    attention = model.attention_pattern(clean_tokens)
    plt.figure(figsize=(5.2, 4.5))
    plt.imshow(attention.numpy(), cmap="magma", aspect="auto")
    plt.colorbar(label="attention weight")
    plt.xlabel("attended token")
    plt.ylabel("query token")
    plt.xticks(range(len(tokens)), tokens)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f"{backend.upper()} induction attention pattern")
    plt.tight_layout()
    plt.savefig(attention_path, dpi=180)
    plt.close()

    recoveries = torch.tensor([float(item["recovery"].max()) for item in results])
    case_summaries = [_case_summary(item) for item in results]
    summary = {
        "backend": backend,
        "n_cases": len(results),
        "n_skipped_cases": len(skipped_cases),
        "clean_prompt": hero["case"]["clean_prompt"],
        "corrupt_prompt": hero["case"]["corrupt_prompt"],
        "correct_token": hero["case"]["correct_token"],
        "incorrect_token": hero["case"]["incorrect_token"],
        "clean_logit_diff": hero["clean_score"],
        "corrupt_logit_diff": hero["corrupt_score"],
        "mean_best_patch_recovery": float(recoveries.mean()),
        "median_best_patch_recovery": float(recoveries.median()),
        "mean_heatmap": "assets/patching_heatmap_mean.png",
        "cases": case_summaries,
        "skipped_cases": [
            {
                "clean_prompt": item["case"]["clean_prompt"],
                "reason": item["reason"],
                "clean_logit_diff": item.get("clean_score"),
                "corrupt_logit_diff": item.get("corrupt_score"),
            }
            for item in skipped_cases
        ],
        "best_patch": {
            "layer": hero["best_layer"],
            "position": hero["best_position"],
            "token": tokens[hero["best_position"]],
            "recovery": float(recovery[hero["best_layer"], hero["best_position"]]),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if backend == "toy":
        interpretation = (
            "Interpretation: in this controlled toy setup, late residual-stream activations at the final "
            "position carry the copy signal needed to recover the held-out source token."
        )
    else:
        interpretation = (
            "Interpretation: GPT-2 patching over a small repeated-text prompt batch finds residual-stream "
            "sites that shift the target logit difference toward the clean prompt behavior. This is a causal "
            "localization result, not yet a complete circuit explanation."
        )

    markdown_path.write_text(
        "# Activation Patching Case Study\n\n"
        f"Evaluated cases: `{summary['n_cases']}`\n\n"
        f"Clean prompt: `{summary['clean_prompt']}` predicts `{summary['correct_token']}`.\n\n"
        f"Corrupt prompt: `{summary['corrupt_prompt']}` suppresses `{summary['correct_token']}` "
        f"in favor of `{summary['incorrect_token']}`.\n\n"
        f"Best patch: layer `{summary['best_patch']['layer']}`, position `{summary['best_patch']['position']}` "
        f"(`{summary['best_patch']['token']}`), recovering `{summary['best_patch']['recovery']:.2f}` "
        "of the clean logit difference.\n\n"
        f"Mean best-patch recovery across cases: `{summary['mean_best_patch_recovery']:.2f}`.\n\n"
        "## Evaluated Cases\n\n"
        "| Clean prompt | Corrupt prompt | Clean logit diff | Corrupt logit diff | Best site | Recovery |\n"
        "| --- | --- | ---: | ---: | --- | ---: |\n"
        + "\n".join(
            "| "
            f"`{case['clean_prompt']}` | "
            f"`{case['corrupt_prompt']}` | "
            f"{case['clean_logit_diff']:.2f} | "
            f"{case['corrupt_logit_diff']:.2f} | "
            f"L{case['best_layer']} P{case['best_position']} `{case['best_token']}` | "
            f"{case['best_recovery']:.2f} |"
            for case in case_summaries
        )
        + "\n\n"
        + (
            "## Skipped Cases\n\n"
            + "\n".join(
                f"- `{case['clean_prompt']}`: {case['reason']} "
                f"(clean `{_format_float(case.get('clean_logit_diff'))}`, corrupt `{_format_float(case.get('corrupt_logit_diff'))}`)"
                for case in summary["skipped_cases"]
            )
            + "\n\n"
            if summary["skipped_cases"]
            else ""
        )
        +
        f"{interpretation}\n",
        encoding="utf-8",
    )
    return {
        "heatmap": heatmap_path,
        "mean_heatmap": mean_heatmap_path,
        "attention": attention_path,
        "summary": json_path,
        "report": markdown_path,
    }
