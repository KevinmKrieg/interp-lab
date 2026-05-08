# interp-lab

`interp-lab` is a small mechanistic interpretability workbench for transformer language models. The default path is intentionally local-first: it ships with a deterministic toy transformer-like backend so the full pipeline can run without downloading model weights. A Hugging Face GPT-2 backend is also available for real-model activation patching and feature experiments.

The project is organized around three resume-visible artifacts:

1. **Causal patching heatmaps** that show where clean activations restore a model behavior.
2. **Sparse autoencoder feature cards** that summarize learned activation features with top contexts.
3. **A tiny NLA-inspired text bottleneck** that verbalizes activations with constrained natural-language labels and reconstructs vectors from those labels.

The NLA module is an inspired toy replica, not a reproduction of Anthropic's RL-based Natural Language Autoencoder training setup.

## Quickstart

```bash
python3 -m pip install -e ".[dev]"
collect-activations
run-patching
train-sae
analyze-features
train-nla-toy
serve-browser --build-only
pytest
```

Open `reports/index.html` after running the commands, or launch a local browser:

```bash
serve-browser --port 8000
```

## Real GPT-2 Backend

The real-model path uses Hugging Face `AutoModelForCausalLM` hooks. It may download GPT-2 weights the first time it runs.

```bash
python3 -m pip install -e ".[models,dev]"
collect-activations --config configs/gpt2_induction.yaml
run-patching --config configs/gpt2_induction.yaml
train-sae --config configs/gpt2_induction.yaml
analyze-features --config configs/gpt2_induction.yaml
train-nla-toy --config configs/gpt2_induction.yaml
serve-browser --config configs/gpt2_induction.yaml --build-only
```

## Example Outputs

### Activation patching

![Activation patching heatmap](reports/assets/patching_heatmap.png)

The heatmap shows logit-difference recovery when clean residual-stream activations are patched into a corrupted prompt. In the toy setup, late final-position activations carry the copy signal needed to recover the held-out source token.

### Attention pattern

![Attention pattern](reports/assets/attention_pattern.png)

The toy induction head attends from the prediction position back to the earlier source token.

### SAE feature activity

![SAE feature histogram](reports/assets/sae_feature_histogram.png)

SAE feature cards are generated in [`reports/sae_feature_cards.md`](reports/sae_feature_cards.md).

### Tiny NLA-inspired bottleneck

![NLA reconstruction](reports/assets/nla_reconstruction.png)

The NLA-inspired module maps an activation to constrained text labels, then reconstructs the activation from the text bottleneck. Example card: [`reports/nla_toy_card.md`](reports/nla_toy_card.md).

## CLI

| Command | Purpose |
| --- | --- |
| `collect-activations` | Cache residual-stream activations for prompt examples. |
| `run-patching` | Run clean/corrupt activation patching and render heatmaps. |
| `train-sae` | Train a sparse autoencoder on cached activations. |
| `analyze-features` | Generate feature cards and activation histograms. |
| `train-nla-toy` | Train the constrained activation-to-text-to-activation toy experiment. |
| `serve-browser` | Serve or build a static report browser. |

## Current Experiment Summary

| Artifact | What it demonstrates | Output |
| --- | --- | --- |
| Patching heatmap | Causal localization of a copy signal | `reports/assets/patching_heatmap.png` |
| Attention pattern | A readable induction-style attention pattern | `reports/assets/attention_pattern.png` |
| SAE cards | Sparse features with top activating contexts | `reports/sae_feature_cards.md` |
| Tiny NLA card | Human-readable activation bottleneck and reconstruction metric | `reports/nla_toy_card.md` |

## Project Layout

```text
configs/           Reproducible experiment configs
src/interp_lab/    Package code and CLI implementations
tests/             Unit and smoke-style tests
reports/           Small checked-in figures and Markdown reports
artifacts/         Large reproducible caches/checkpoints, ignored by git
```

## Roadmap

- Add a TransformerLens backend for richer named activation sites and standard circuit-analysis utilities.
- Add richer circuit case studies with clean/corrupt prompt datasets.
- Add a browser UI for filtering SAE features by token, prompt, and activation range.
- Compare the tiny NLA bottleneck against SAE-only and nearest-neighbor label baselines.
