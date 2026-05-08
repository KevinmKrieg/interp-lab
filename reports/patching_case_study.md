# Activation Patching Case Study

Clean prompt: `A B C A B` predicts `C`.

Corrupt prompt: `A B C A D` suppresses `C` in favor of `D`.

Best patch: layer `5`, position `4` (`B`), recovering `1.55` of the clean logit difference.

Interpretation: in this controlled toy setup, late residual-stream activations at the final position carry the copy signal needed to recover the held-out source token.
