# Activation Patching Case Study

Evaluated cases: `6`

Clean prompt: `Mary had a little lamb. Mary had a little` predicts ` lamb`.

Corrupt prompt: `Mary had a little lamb. Mary had a small` suppresses ` lamb` in favor of ` house`.

Best patch: layer `0`, position `9` (` little`), recovering `1.00` of the clean logit difference.

Mean best-patch recovery across cases: `1.00`.

## Evaluated Cases

| Clean prompt | Corrupt prompt | Clean logit diff | Corrupt logit diff | Best site | Recovery |
| --- | --- | ---: | ---: | --- | ---: |
| `Mary had a little lamb. Mary had a little` | `Mary had a little lamb. Mary had a small` | 7.41 | 5.93 | L0 P9 ` little` | 1.00 |
| `To be or not to be. To be or not to` | `To be or not to be. To be or not for` | 9.41 | -2.86 | L0 P11 ` to` | 1.00 |
| `The quick brown fox jumps. The quick brown fox` | `The quick brown fox jumps. The quick brown dog` | 4.45 | 2.91 | L0 P9 ` fox` | 1.00 |
| `New York City is busy. New York City is` | `New York City is busy. New York City was` | 2.46 | -1.44 | L0 P9 ` is` | 1.00 |
| `A B C D. A B C` | `A B C D. A B X` | 5.66 | -1.26 | L0 P7 ` C` | 1.00 |
| `red blue green yellow. red blue green` | `red blue green yellow. red blue purple` | 2.78 | 1.38 | L0 P7 ` green` | 1.00 |

## Skipped Cases

- `Once upon a time there was. Once upon a time`: clean logit margin does not exceed corrupt margin (clean `2.94`, corrupt `4.43`)

Interpretation: GPT-2 patching over a small repeated-text prompt batch finds residual-stream sites that shift the target logit difference toward the clean prompt behavior. This is a causal localization result, not yet a complete circuit explanation.
