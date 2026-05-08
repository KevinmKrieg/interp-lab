from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PatchSpec:
    layer: int
    position: int
    value: torch.Tensor


class HuggingFaceCausalLM:
    """Minimal hooked interface for GPT-style Hugging Face causal LMs."""

    def __init__(self, config: dict):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install the optional model dependencies with `pip install -e '.[models]'`.") from exc

        model_config = config["model"]
        model_name = model_config.get("name", "gpt2")
        self.device = torch.device(model_config.get("device", "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
        self.model.to(self.device)
        self.model.eval()
        self.n_layers = int(self.model.config.n_layer)
        self.d_model = int(self.model.config.n_embd)
        self.vocab = list(self.tokenizer.get_vocab().keys())
        self._last_attentions: tuple[torch.Tensor, ...] | None = None

    def to_tokens(self, prompt: str) -> torch.Tensor:
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"][0].to(self.device)

    def token_id(self, token: str) -> int:
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"`{token}` maps to {len(ids)} tokens; use a single-token string.")
        return int(ids[0])

    def token_label(self, token_id: int) -> str:
        text = self.tokenizer.decode([int(token_id)])
        return text.replace("\n", "\\n") or f"<{token_id}>"

    def token_labels(self, tokens: torch.Tensor) -> list[str]:
        return [self.token_label(int(token)) for token in tokens.detach().cpu()]

    def to_string(self, tokens: torch.Tensor | list[int]) -> str:
        return self.tokenizer.decode([int(token) for token in tokens])

    def run_with_cache(self, tokens: torch.Tensor, patch: PatchSpec | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = tokens.unsqueeze(0).to(self.device)
        handles = []

        if patch is not None:
            patch_value = patch.value.to(self.device)

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden.clone()
                hidden[:, patch.position, :] = patch_value
                if isinstance(output, tuple):
                    return (hidden, *output[1:])
                return hidden

            handles.append(self.model.transformer.h[patch.layer].register_forward_hook(hook))

        with torch.no_grad():
            output = self.model(input_ids, output_hidden_states=True, output_attentions=True)

        for handle in handles:
            handle.remove()

        self._last_attentions = output.attentions
        cache = torch.stack([hidden[0].detach().cpu() for hidden in output.hidden_states[1:]])
        logits = output.logits[0, -1].detach().cpu()
        return logits, cache

    def attention_pattern(self, _tokens: torch.Tensor, head: int = 0) -> torch.Tensor:
        if self._last_attentions is None:
            raise RuntimeError("Run the model before requesting attention patterns.")
        layer_attention = self._last_attentions[-1][0]
        if layer_attention is None:
            raise RuntimeError("This Hugging Face model did not return attention tensors.")
        return layer_attention[head % layer_attention.shape[0]].detach().cpu()
