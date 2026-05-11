# Activation Patching Case Study

Evaluated cases: `1`

Clean prompt: `A B C A B` predicts `C`.

Corrupt prompt: `A B C A D` suppresses `C` in favor of `D`.

Best patch: layer `5`, position `4` (`B`), recovering `1.55` of the clean logit difference.

Mean best-patch recovery across cases: `1.55`.

## Evaluated Cases

| Clean prompt | Corrupt prompt | Clean logit diff | Corrupt logit diff | Best site | Recovery |
| --- | --- | ---: | ---: | --- | ---: |
| `A B C A B` | `A B C A D` | 50.87 | 18.29 | L5 P4 `B` | 1.55 |

Interpretation: in this controlled toy setup, late residual-stream activations at the final position carry the copy signal needed to recover the held-out source token.
