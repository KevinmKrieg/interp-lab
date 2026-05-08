# Activation Patching Case Study

Clean prompt: `A B C A B` predicts ` C`.

Corrupt prompt: `A B C A D` suppresses ` C` in favor of ` D`.

Best patch: layer `11`, position `4` (` B`), recovering `6.35` of the clean logit difference.

Interpretation: patching this real-model residual-stream site strongly shifts the target logit difference toward the clean prompt behavior. This is a causal localization result, not yet a complete circuit explanation.
