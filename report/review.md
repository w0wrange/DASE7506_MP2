# Report outline and claim/evidence review

Outline: research target and history; compact verifier and trace rule; training and
selection; matched validation comparisons/ablations; failures, resources and limits.

| Dimension | Review question and resolution |
|---|---|
| Contribution | Are we claiming a new verification paradigm? No. Established verifier selection is cited; contribution is implementation and constrained analysis. |
| Clarity | Can the release be reproduced? Architecture, loss, scalers, seeds, folds, grid, environment and exact commands are supplied. |
| Experimental strength | Does +2 validation prove stable improvement? No. Paired counts, descriptive p=0.625 and adaptive-selection caveat are explicit. |
| Completeness | Are controlled ablations confused with the final model? No. Earlier 63-feature/23-epoch settings are labeled; final trace ablation is separate. |
| Design soundness | Does arithmetic consistency prove semantic correctness? No. Pair-count failure and formatted-trace miss are analyzed. |

Claim: Final formal score is 394/467. Evidence: unchanged official evaluator JSON and
frozen checkpoint. Status: reproducible numerical result, with a known test-use
restriction conflict; it is not a clean held-out generalization claim.

Claim: Structure and comparison help in the controlled earlier MLP. Evidence:
experiments/controlled-ablations.json, matched seed/configuration. Status: supported
for those settings; not evidence of universal improvement or final-model stability.

Claim: Final trace rule adds two validation successes. Evidence: validation paired
3 gains/1 loss, training OOF 12/1. Status: supported, small, selection-adjusted
inference unavailable.

Claim: Assets are below 16 MiB. Evidence: uncompressed inventory. Status: supported.
Claim: Reference CPU evaluation finishes within 30 seconds. Evidence: unavailable.
Status: unverified; report explicitly limits timing claims to local/proxy hardware.

Claim: Experiment fully follows the test-use rule. Evidence: contrary historical
test feedback and stopping objective. Status: false; not claimed. Instructor review
is required, and a clean directory or training replay cannot restore that boundary.
