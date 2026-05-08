# Activation Patching Case Study

Evaluated cases: `6`

Clean prompt: `Mary had a little lamb. Mary had a little` predicts ` lamb`.

Corrupt prompt: `Mary had a little lamb. Mary had a small` suppresses ` lamb` in favor of ` house`.

Best patch: layer `0`, position `9` (` little`), recovering `1.00` of the clean logit difference.

Mean best-patch recovery across cases: `1.00`.

Interpretation: GPT-2 patching over a small repeated-text prompt batch finds residual-stream sites that shift the target logit difference toward the clean prompt behavior. This is a causal localization result, not yet a complete circuit explanation.
