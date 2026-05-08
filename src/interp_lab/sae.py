from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, n_features: int, k_sparse: int = 8):
        super().__init__()
        self.k_sparse = k_sparse
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.encoder(x))
        if self.k_sparse < features.shape[-1]:
            values, indices = torch.topk(features, k=self.k_sparse, dim=-1)
            sparse = torch.zeros_like(features)
            sparse.scatter_(-1, indices, values)
            return sparse
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return self.decode(features), features


def train_sae(config: dict) -> Path:
    cache = torch.load(Path(config["outputs"]["artifacts_dir"]) / "activation_cache.pt", weights_only=False)
    raw_activations = cache["activations"]
    activation_mean = raw_activations.mean(dim=0, keepdim=True)
    activation_std = raw_activations.std(dim=0, keepdim=True).clamp_min(1e-6)
    activations = (raw_activations - activation_mean) / activation_std
    torch.manual_seed(int(config["seed"]))
    model = SparseAutoencoder(d_model=activations.shape[-1], n_features=int(config["training"]["sae_features"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = int(config["training"]["sae_steps"])

    for step in range(steps):
        indices = torch.randint(0, activations.shape[0], (min(64, activations.shape[0]),))
        batch = activations[indices]
        reconstruction, features = model(batch)
        loss = F.mse_loss(reconstruction, batch) + 1e-3 * features.abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    output = Path(config["outputs"]["artifacts_dir"]) / "sae.pt"
    with torch.no_grad():
        reconstruction, features = model(activations)
        metrics = {
            "mse": float(F.mse_loss(reconstruction, activations)),
            "mean_l0": float((features > 1e-4).float().sum(dim=-1).mean()),
        }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "d_model": activations.shape[-1],
            "n_features": int(config["training"]["sae_features"]),
            "k_sparse": model.k_sparse,
            "activation_mean": activation_mean,
            "activation_std": activation_std,
            "metrics": metrics,
        },
        output,
    )
    return output


def analyze_features(config: dict, top_k: int = 4) -> dict[str, Path]:
    artifacts = Path(config["outputs"]["artifacts_dir"])
    reports = Path(config["outputs"]["reports_dir"])
    cache = torch.load(artifacts / "activation_cache.pt", weights_only=False)
    checkpoint = torch.load(artifacts / "sae.pt", weights_only=False)
    sae = SparseAutoencoder(checkpoint["d_model"], checkpoint["n_features"], checkpoint.get("k_sparse", 8))
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()

    with torch.no_grad():
        activations = (cache["activations"] - checkpoint["activation_mean"]) / checkpoint["activation_std"]
        features = sae.encode(activations)
    max_values, max_indices = features.max(dim=0)
    selected = torch.topk(max_values, k=min(20, features.shape[1])).indices.tolist()
    cards = []

    for feature_id in selected:
        values = features[:, feature_id]
        top_indices = torch.topk(values, k=min(top_k, values.numel())).indices.tolist()
        examples = []
        token_counts: dict[str, int] = {}
        for index in top_indices:
            metadata = cache["metadata"][index]
            token = metadata.get("token_text", str(metadata["token_id"]))
            token_counts[token] = token_counts.get(token, 0) + 1
            examples.append(
                {
                    "activation": float(values[index]),
                    "token": token,
                    "position": metadata["position"],
                    "prompt": metadata["prompt"],
                }
            )
        label_token = max(token_counts, key=token_counts.get)
        label = f"{label_token}-token residual feature"
        cards.append({"feature_id": feature_id, "label": label, "max_activation": float(max_values[feature_id]), "examples": examples})

    json_path = reports / "sae_feature_cards.json"
    md_path = reports / "sae_feature_cards.md"
    plot_path = reports / "assets" / "sae_feature_histogram.png"
    json_path.write_text(json.dumps({"metrics": checkpoint["metrics"], "features": cards}, indent=2), encoding="utf-8")

    lines = ["# SAE Feature Cards\n", f"SAE reconstruction MSE: `{checkpoint['metrics']['mse']:.4f}`\n", f"Mean active features: `{checkpoint['metrics']['mean_l0']:.2f}`\n"]
    for card in cards[:8]:
        lines.append(f"## Feature {card['feature_id']}: {card['label']}\n")
        lines.append(f"Max activation: `{card['max_activation']:.3f}`\n")
        for example in card["examples"]:
            lines.append(f"- `{example['prompt']}` at position `{example['position']}` token `{example['token']}`: `{example['activation']:.3f}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    plt.figure(figsize=(7, 4))
    plt.hist(features.flatten().detach().numpy(), bins=40, color="#287c8e")
    plt.title("SAE feature activation distribution")
    plt.xlabel("activation")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    return {"json": json_path, "markdown": md_path, "histogram": plot_path}
