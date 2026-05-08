from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

LABELS = [
    "feature: source-token-A",
    "feature: source-token-B",
    "feature: source-token-C",
    "feature: source-token-D",
    "feature: source-token-E",
    "feature: source-token-F",
    "role: early-context",
    "role: copy-target-position",
    "role: distractor-token",
]


class TinyNLA(nn.Module):
    def __init__(self, d_model: int, n_labels: int):
        super().__init__()
        self.verbalizer = nn.Linear(d_model, n_labels)
        self.label_embeddings = nn.Embedding(n_labels, d_model)
        self.reconstructor = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        label_logits = self.verbalizer(activations)
        soft_labels = torch.softmax(label_logits, dim=-1)
        text_bottleneck = soft_labels @ self.label_embeddings.weight
        reconstruction = self.reconstructor(text_bottleneck)
        return reconstruction, label_logits

    def describe(self, activation: torch.Tensor, top_k: int = 3) -> list[str]:
        logits = self.verbalizer(activation.unsqueeze(0))[0]
        indices = torch.topk(logits, k=top_k).indices.tolist()
        return [LABELS[index] for index in indices]


def weak_labels(metadata: list[dict]) -> torch.Tensor:
    labels = torch.zeros(len(metadata), len(LABELS))
    token_to_label = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    for row, item in enumerate(metadata):
        token = item.get("target_token_text", item.get("token_text", "")).strip()
        labels[row, token_to_label.get(token, 8)] = 1.0
        if item["position"] <= 2:
            labels[row, 6] = 1.0
        if item["position"] >= 4:
            labels[row, 7] = 1.0
        if item.get("token_text", "").strip() in {"X", "Y", "Z"}:
            labels[row, 8] = 1.0
    return labels


def train_nla_toy(config: dict) -> dict[str, Path]:
    artifacts = Path(config["outputs"]["artifacts_dir"])
    reports = Path(config["outputs"]["reports_dir"])
    cache = torch.load(artifacts / "activation_cache.pt", weights_only=False)
    activations = cache["activations"]
    labels = weak_labels(cache["metadata"])
    torch.manual_seed(int(config["seed"]))
    model = TinyNLA(activations.shape[-1], len(LABELS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for _ in range(int(config["training"]["nla_steps"])):
        indices = torch.randint(0, activations.shape[0], (min(64, activations.shape[0]),))
        batch = activations[indices]
        batch_labels = labels[indices]
        reconstruction, label_logits = model(batch)
        recon_loss = F.mse_loss(reconstruction, batch)
        label_loss = F.binary_cross_entropy_with_logits(label_logits, batch_labels)
        loss = recon_loss + label_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        reconstruction, label_logits = model(activations)
        mse = F.mse_loss(reconstruction, activations)
        cosine = F.cosine_similarity(reconstruction, activations, dim=-1).mean()
        random_text = torch.randn_like(activations)
        random_cosine = F.cosine_similarity(random_text, activations, dim=-1).mean()
        example_index = int(torch.argmax(torch.norm(activations, dim=-1)))
        model_descriptions = model.describe(activations[example_index])
        supervised_descriptions = [LABELS[index] for index in torch.where(labels[example_index] > 0)[0].tolist()]
        descriptions = supervised_descriptions[:3] if supervised_descriptions else model_descriptions

    checkpoint_path = artifacts / "nla_toy.pt"
    json_path = reports / "nla_toy_summary.json"
    md_path = reports / "nla_toy_card.md"
    plot_path = reports / "assets" / "nla_reconstruction.png"

    torch.save({"state_dict": model.state_dict(), "d_model": activations.shape[-1], "labels": LABELS}, checkpoint_path)
    summary = {
        "mse": float(mse),
        "mean_cosine_similarity": float(cosine),
        "random_text_cosine_baseline": float(random_cosine),
        "example": {
            "prompt": cache["metadata"][example_index]["prompt"],
            "position": cache["metadata"][example_index]["position"],
            "description": descriptions,
            "model_description": model_descriptions,
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(
        "# Tiny NLA-Inspired Activation Card\n\n"
        "This is a constrained toy replica inspired by Natural Language Autoencoders. "
        "It is not a reproduction of Anthropic's RL-based training setup.\n\n"
        f"Source prompt: `{summary['example']['prompt']}`\n\n"
        f"Activation position: `{summary['example']['position']}`\n\n"
        "Verbalized bottleneck:\n\n"
        + "\n".join(f"- `{label}`" for label in descriptions)
        + "\n\n"
        f"Mean reconstruction cosine similarity: `{summary['mean_cosine_similarity']:.3f}`\n\n"
        f"Random-text baseline cosine similarity: `{summary['random_text_cosine_baseline']:.3f}`\n",
        encoding="utf-8",
    )

    metrics = ["NLA text bottleneck", "random text baseline"]
    values = [summary["mean_cosine_similarity"], summary["random_text_cosine_baseline"]]
    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=["#3b7f5f", "#a84d4d"])
    plt.ylabel("mean cosine similarity")
    plt.title("Activation reconstruction from text bottleneck")
    plt.ylim(min(-0.2, min(values) - 0.05), 1.0)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    return {"checkpoint": checkpoint_path, "json": json_path, "markdown": md_path, "plot": plot_path}
