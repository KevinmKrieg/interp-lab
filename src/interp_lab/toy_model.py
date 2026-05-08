from __future__ import annotations

from dataclasses import dataclass

import torch


VOCAB = ["<pad>", "A", "B", "C", "D", "E", "F", "X", "Y", "Z", "."]
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}


@dataclass(frozen=True)
class PatchSpec:
    layer: int
    position: int
    value: torch.Tensor


class ToyHookedTransformer:
    """A deterministic transformer-like model for offline interp smoke tests.

    The model creates residual-stream activations with an explicit copy circuit:
    token information from position 2 is written into late-layer activations at
    the final position. Patching those sites restores the clean next-token logit.
    """

    def __init__(self, n_layers: int = 6, d_model: int = 32, seed: int = 7):
        self.n_layers = n_layers
        self.d_model = d_model
        self.vocab = VOCAB
        generator = torch.Generator().manual_seed(seed)
        self.token_embed = torch.randn(len(VOCAB), d_model, generator=generator)
        self.pos_embed = torch.randn(32, d_model, generator=generator) * 0.15
        self.layer_embed = torch.randn(n_layers, d_model, generator=generator) * 0.08
        self.copy_direction = torch.randn(d_model, generator=generator)
        self.copy_direction = self.copy_direction / self.copy_direction.norm()
        self.unembed = self.token_embed.clone()

    def to_tokens(self, prompt: str) -> torch.Tensor:
        ids = [TOKEN_TO_ID[token] for token in prompt.split()]
        return torch.tensor(ids, dtype=torch.long)

    def to_string(self, tokens: torch.Tensor | list[int]) -> str:
        return " ".join(ID_TO_TOKEN[int(token)] for token in tokens)

    def token_id(self, token: str) -> int:
        return TOKEN_TO_ID[token]

    def token_label(self, token_id: int) -> str:
        return ID_TO_TOKEN[int(token_id)]

    def token_labels(self, tokens: torch.Tensor) -> list[str]:
        return [self.token_label(int(token)) for token in tokens.detach().cpu()]

    def run_with_cache(self, tokens: torch.Tensor, patch: PatchSpec | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 1:
            raise ValueError("ToyHookedTransformer expects a 1D token sequence.")
        seq_len = tokens.shape[0]
        cache = torch.empty(self.n_layers, seq_len, self.d_model)
        source_token = tokens[2] if seq_len > 2 else tokens[-1]
        source_signal = self.token_embed[source_token] + 1.4 * self.copy_direction

        for layer in range(self.n_layers):
            depth = layer / max(1, self.n_layers - 1)
            for position, token in enumerate(tokens):
                activation = self.token_embed[token] + self.pos_embed[position] + self.layer_embed[layer]
                if position == 2:
                    activation = activation + (0.6 + depth) * source_signal
                if position == seq_len - 1 and layer >= self.n_layers // 2:
                    activation = activation + (1.0 + depth) * source_signal
                cache[layer, position] = activation

        if patch is not None:
            cache[patch.layer, patch.position] = patch.value

        final_residual = cache[-1, -1].clone()
        if patch is not None and patch.position == seq_len - 1:
            final_residual = cache[patch.layer, patch.position] + 0.35 * cache[-1, -1]
        logits = final_residual @ self.unembed.T
        return logits, cache

    def attention_pattern(self, tokens: torch.Tensor, head: int = 0) -> torch.Tensor:
        seq_len = tokens.shape[0]
        pattern = torch.full((seq_len, seq_len), 0.04)
        pattern = torch.tril(pattern)
        for position in range(seq_len):
            pattern[position, : position + 1] /= pattern[position, : position + 1].sum()
        if seq_len > 3:
            pattern[-1] = 0.02
            pattern[-1, 2] = 0.78
            pattern[-1, -2] = 0.12
            pattern[-1, : seq_len] /= pattern[-1, : seq_len].sum()
        if head % 2 == 1:
            pattern = torch.roll(pattern, shifts=1, dims=1)
            pattern = pattern / pattern.sum(dim=-1, keepdim=True)
        return pattern
