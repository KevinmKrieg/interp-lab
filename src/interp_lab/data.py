from __future__ import annotations

import random

import torch

from interp_lab.toy_model import TOKEN_TO_ID, ToyHookedTransformer


def make_induction_pairs(n_examples: int, seed: int = 7) -> list[dict[str, str]]:
    rng = random.Random(seed)
    source_tokens = ["A", "B", "C", "D", "E", "F"]
    distractors = ["X", "Y", "Z"]
    pairs = []
    for _ in range(n_examples):
        a, b, c = rng.sample(source_tokens, 3)
        corrupt = rng.choice([token for token in distractors if token != c])
        incorrect = rng.choice([token for token in source_tokens + distractors if token not in {c, corrupt}])
        pairs.append(
            {
                "clean_prompt": f"{a} {b} {c} {a} {b}",
                "corrupt_prompt": f"{a} {b} {c} {a} {incorrect}",
                "correct_token": f" {c}",
                "incorrect_token": f" {incorrect}",
            }
        )
    return pairs


def make_text_induction_pairs() -> list[dict[str, str]]:
    return [
        {
            "clean_prompt": "Mary had a little lamb. Mary had a little",
            "corrupt_prompt": "Mary had a little lamb. Mary had a small",
            "correct_token": " lamb",
            "incorrect_token": " house",
        },
        {
            "clean_prompt": "To be or not to be. To be or not to",
            "corrupt_prompt": "To be or not to be. To be or not for",
            "correct_token": " be",
            "incorrect_token": " the",
        },
        {
            "clean_prompt": "The quick brown fox jumps. The quick brown fox",
            "corrupt_prompt": "The quick brown fox jumps. The quick brown dog",
            "correct_token": " jumps",
            "incorrect_token": " runs",
        },
        {
            "clean_prompt": "Once upon a time there was. Once upon a time",
            "corrupt_prompt": "Once upon a time there was. Once upon a day",
            "correct_token": " there",
            "incorrect_token": " the",
        },
        {
            "clean_prompt": "New York City is busy. New York City is",
            "corrupt_prompt": "New York City is busy. New York City was",
            "correct_token": " busy",
            "incorrect_token": " the",
        },
        {
            "clean_prompt": "A B C D. A B C",
            "corrupt_prompt": "A B C D. A B X",
            "correct_token": " D",
            "incorrect_token": " Y",
        },
        {
            "clean_prompt": "red blue green yellow. red blue green",
            "corrupt_prompt": "red blue green yellow. red blue purple",
            "correct_token": " yellow",
            "incorrect_token": " orange",
        },
    ]


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
