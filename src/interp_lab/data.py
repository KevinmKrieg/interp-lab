from __future__ import annotations

import random

import torch

from interp_lab.toy_model import TOKEN_TO_ID, ToyHookedTransformer


def make_copy_prompts(n_examples: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    source_tokens = ["A", "B", "C", "D", "E", "F"]
    distractors = ["X", "Y", "Z"]
    prompts = []
    for _ in range(n_examples):
        a, b, c = rng.sample(source_tokens, 3)
        d = rng.choice([token for token in distractors if token != c])
        prompts.append(f"{a} {b} {c} {a} {b} {d}")
    return prompts


def encode_prompts(model: ToyHookedTransformer, prompts: list[str]) -> torch.Tensor:
    tokens = [model.to_tokens(prompt) for prompt in prompts]
    return torch.stack(tokens)


def target_token_for_prompt(prompt: str) -> int:
    return TOKEN_TO_ID[prompt.split()[2]]
