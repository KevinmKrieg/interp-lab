import torch

from interp_lab.patching import logit_diff
from interp_lab.models import make_model
from interp_lab.toy_model import PatchSpec


def test_activation_cache_shapes():
    config = {"seed": 7, "model": {"backend": "toy", "n_layers": 4, "d_model": 16}}
    model = make_model(config)
    tokens = model.to_tokens("A B C A D")
    logits, cache = model.run_with_cache(tokens)
    assert logits.shape == (len(model.vocab),)
    assert cache.shape == (4, 5, 16)


def test_final_position_patch_improves_clean_logit_diff():
    config = {"seed": 7, "model": {"backend": "toy", "n_layers": 6, "d_model": 32}}
    model = make_model(config)
    clean = model.to_tokens("A B C A B")
    corrupt = model.to_tokens("A B C A D")
    clean_logits, clean_cache = model.run_with_cache(clean)
    corrupt_logits, _ = model.run_with_cache(corrupt)
    patch = PatchSpec(layer=5, position=4, value=clean_cache[5, 4])
    patched_logits, _ = model.run_with_cache(corrupt, patch=patch)

    corrupt_score = logit_diff(corrupt_logits, model.token_id("C"), model.token_id("D"))
    patched_score = logit_diff(patched_logits, model.token_id("C"), model.token_id("D"))
    clean_score = logit_diff(clean_logits, model.token_id("C"), model.token_id("D"))
    assert abs(clean_score - patched_score) < abs(clean_score - corrupt_score)
