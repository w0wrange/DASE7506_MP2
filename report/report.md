# Compact Candidate Verification with Arithmetic Trace Evidence

DASE7506 Project 2 | Student ID 3036802038 | 31 October 2026

## Abstract

This project studies whether comparing candidate solutions and checking explicit arithmetic can improve selection of a correct final numerical answer. A compact verifier combines logistic regression on supplied 1,024-dimensional representations with a shared MLP using relative candidate representations and 75 structural/arithmetic features. An input-only residual rule rewards existing candidate answers supported by bounded arithmetic recomputation. On the official validation bank, the frozen method selects 743/798 correct answers (93.11%), including 300/355 disagreement questions (84.51%), versus 688/798 (86.22%) and 245/355 (69.01%) for parsed majority voting. Removing the residual rule reduces validation accuracy by two questions. The final official test result is 394/467 (84.3683%). Inference assets occupy 1.154 MiB uncompressed. These results describe this course bank, with the experimental-history limitation below.

## Experimental-history and eligibility disclosure

Development included repeated aggregate test-reference evaluations and an explicit test-informed stopping objective. This conflicts with the course rule that test references are for final scoring and reproduction only. Consequently, the final test score is an exploratory development-reference result, not an unbiased evaluation on an untouched holdout. Cleaning the repository cannot repair that history; instructor review is required for eligibility. Model parameters were fitted on training labels, and the final residual grid used training OOF and validation evidence, but broader method exploration and stopping occurred with test feedback known.

An earlier broad source search also accidentally returned test-reference text. Its incident note is retained. Earlier numerical exploratory test results were not persisted at the user's request; a complete historical query log is unavailable and has not been invented. Only the final formal test evaluation JSON is retained with authorization. No test examples or test error analysis are used in this report. See DISCLOSURE.md for the full statement.

## 1. Research question and evaluation target

The research question is whether candidate comparison, answer agreement and explicit arithmetic evidence improve final-answer selection over majority voting and independent candidate scoring. The hypothesis is that shared representations expose relative plausibility, while arithmetic evidence can correct a small subset of numerical slips that semantic embeddings miss. This is a project-specific design and empirical comparison; verifier-based selection is established prior work [1].

The course bank contains 4,755 training, 798 validation and 467 test questions, each with four fixed solutions and representations of shape [N,4,1024]. Every question has at least one correct parsed answer, and several candidates can be correct. We label all matching candidates positive using the unchanged official rational-number parser. The disagreement subset contains at least two distinct valid parsed answers: 2,114 training and 355 validation questions; all test questions satisfy this condition. This subset is more diagnostic than the overall metric, which includes easy answer-consensus groups.

<!-- pagebreak -->

## 2. Method

**Parser-aware selection.** Invalid parsed answers are masked. If there is only one distinct valid answer, the first candidate with that answer is selected directly. With no valid answers, the fallback is index 0; the released benchmark guarantees a correct parseable candidate. Duplicate correct candidates are all valid choices. Ties use first occurrence. IDs are used only by the official runner for output alignment.

**Linear member.** L2-regularized logistic regression (C=0.1, LBFGS, at most 400 iterations) consumes a standardized supplied embedding for each candidate. Training partition statistics define the mean and standard deviation, with a 0.05 scale floor. Hard-question sample weights are 2.0 and consensus-question weights 0.5. This member provides a compact independent supervised baseline and a complementary fusion signal.

**Shared MLP member.** Each standardized embedding passes through 1024 -> 256 -> 64 layers with ReLU and dropout 0.3. Let h_i denote its output and h_bar the mean across four candidates. A 64-unit ReLU head receives [h_i, h_bar, h_i-h_bar, f_i] and produces one logit. The vector f_i contains 63 base features and 12 arithmetic features. Base features include parsed-answer votes, distinct-answer counts, candidate cosine similarity, numerical overlap with the question, length/format statistics and simple equation consistency. The added features improve normalization of explicit fractions, percentages and arithmetic expressions. Unsupported expressions remain unchecked; no symbolic or language-level proof is attempted.

For the MLP, embedding and feature standardizers are fitted only on the fitting partition and inputs clipped to [-8,8] after scaling. Training uses AdamW (learning rate 0.001, weight decay 0.01), batches of 256 questions, gradient norm cap 5, deterministic PyTorch algorithms and 19 fixed epochs for the final recipe. The loss per question is mean binary cross-entropy plus 0.5 times the negative log probability mass of all correct candidates. The same hard/easy weights are applied at question level. This supports multiple correct answers without arbitrarily choosing one target.

**Probability fusion.** For linear and neural logits s_L and s_N, masked softmax distributions are combined as p = 0.3935807962 softmax(s_L/0.9056050280) + 0.6064192038 softmax(s_N/3.0717703560). The weights and temperatures originate from original training-OOF calibration and were retained when adding arithmetic features. They are not fitted on test labels. The final rule normally chooses the maximum-scoring valid candidate rather than summing all candidates sharing an answer.

**Arithmetic trace residual.** Each candidate is scanned for explicit numerical equalities. A restricted AST evaluator permits constants, signs and +, -, *, / with exact Fraction arithmetic; expressions are bounded to 50 AST nodes, absolute magnitude 10^15 and 256-bit numerator/denominator. Clear mismatches can map an erroneous intermediate number to its recomputed value, then propagate that mapping through later numerical expressions. Zero and explicit question-number replacements are avoided; more than eight errors causes abstention. Display-rounding tolerance and relative tolerance 0.001 limit spurious corrections.

For each existing answer a_i, c_i counts candidate traces proposing a correction to that answer. The released residual is log(max(p_i,10^-8)) + c_i, with no self-taint penalty. It only changes selection among the four supplied answers; it does not generate a new solution or access references. Heuristic numerical substitution can be ambiguous when the same number denotes different quantities, so the rule is evidence rather than a correctness guarantee.

<!-- pagebreak -->

## 3. Experimental design and model selection

Splits are kept at question level so the four candidates never cross fitting and held-out partitions. The original search compared six sparse text configurations, twelve CatBoost configurations and up to twenty-four MLP configurations on a fixed training 80/20 split. Shortlisted settings received five-fold OOF evaluation. Historical search records preserve their settings and outcomes; this release replays the final recipe rather than reconstructing every exploratory query.

Five fixed folds generate training OOF logits for the two released members, using seeds 42+fold. Full-data models use seed 42. This entails twelve fits, including two full fits. Feature transforms are refitted for each training partition. The 19-epoch setting was inherited from earlier training development. Official validation labels are used for method comparison and residual selection, not gradient updates or normalizer fitting. The training helper rejects any request to load test references.

The residual grid contains support bonuses and taint penalties in {0,0.5,1,2,4}, crossed with candidate maximum or answer-mean aggregation: 50 combinations. Configurations must not reduce the base fusion's training-OOF hard count. Eligible configurations maximize validation correct count, breaking ties by OOF correct count and then smaller residual strength. The selected configuration is support=1, penalty=0, candidate maximum. The OOF and validation selectors agreed. These adaptive choices make OOF/validation results descriptive, not pristine estimates for a preregistered hypothesis.

| Final-recipe comparison | Training OOF hard / 2,114 | Validation hard / 355 |
|---|---:|---:|
| Fusion without trace residual | 1,748 (82.69%) | 298 (83.94%) |
| Fusion with trace residual | 1,759 (83.21%) | 300 (84.51%) |
| Corrected / newly wrong | 12 / 1 | 3 / 1 |

The paired exact McNemar descriptive p-values are 0.00342 on training OOF and 0.625 on validation. They are not adjusted for model selection or multiple comparisons. The small validation gain does not establish a reliable general improvement.

Historical three-seed arithmetic experiments provide additional caution. On outer training holdouts, arithmetic fusion averaged 82.43% hard accuracy versus 82.39% for original fusion; on validation, their means were 84.41% and 83.76%. Learned dynamic gates averaged only 81.77-81.96% on outer training holdouts and were not retained. These results support testing arithmetic evidence further but do not prove that extra feature engineering always helps. The released model uses one seed; there is no three-seed replication of the complete final trace-corrected pipeline.

**Background and scope.** Cobbe et al. [1] trained verifiers to rank generated mathematical solutions. Here the candidates and representations are fixed by the course, with no extra pretrained weights or external training data. The contribution is a compact implementation and constrained empirical analysis, not a new general verification paradigm. Accuracy measures agreement with the supplied final numerical reference, not validity of every intermediate step.

<!-- pagebreak -->

## 4. Validation comparisons and controlled ablations

All results in the next table use the same 798-question validation bank and official parsing. Majority ties choose the answer whose first candidate occurs earliest. Supervised methods share parser masking and consensus selection, so gains over the raw first-candidate baseline include that deterministic preprocessing.

| Method | Correct / 798 | Overall % | Hard correct / 355 | Hard % |
|---|---:|---:|---:|---:|
| First candidate | 624 | 78.20 | 195 | 54.93 |
| Parsed majority voting | 688 | 86.22 | 245 | 69.01 |
| Linear embedding classifier | 722 | 90.48 | 279 | 78.59 |
| Arithmetic shared MLP | 739 | 92.61 | 296 | 83.38 |
| Linear + arithmetic MLP fusion | 741 | 92.86 | 298 | 83.94 |
| Final fusion + trace residual | 743 | 93.11 | 300 | 84.51 |

The final method improves hard accuracy by 15.49 percentage points over parsed majority voting. Fusion adds two correct validation choices over the arithmetic MLP; the trace residual adds two further choices over fusion. These comparisons are development-bank outcomes after extensive exploration. Only the final formal test result is retained: **394/467 = 84.36830835117773%**, produced by the unchanged official scorer. Because of prior test feedback, that score must be interpreted with the disclosure on page 1.

Historical controlled ablations isolate inputs using the same seed, split and hyperparameters within each block. The MLP block uses the earlier 63-feature model, width 256, dropout 0.1, learning rate 0.0003 and 23 epochs, rather than the final arithmetic model. Thus it measures the earlier architecture, not the exact final residual pipeline.

| Controlled validation experiment | Hard correct / 355 | Hard % |
|---|---:|---:|
| Earlier shared MLP, seed 42 | 297 | 83.66 |
| Remove relative neural representations only | 292 | 82.25 |
| Remove relative representations and group-dependent features | 292 | 82.25 |
| Remove all structural features | 278 | 78.31 |
| Word text LR: question + final answer | 157 | 44.23 |
| Word text LR: question + full solution | 209 | 58.87 |

The text comparison holds word hashing, L2 C=1 and training data constant. Reading the full text helps this sparse model by 52 hard questions, but it still underperforms the learned embedding verifier. The MLP ablations suggest that structure and candidate comparison matter for that setting; removing structure costs 19 hard questions, whereas removing comparison costs five. Three seeds (42,43,44) of the earlier full MLP give hard accuracies 83.66%, 83.38% and 83.10%, respectively. These limited replications do not establish robustness of every later choice.

No extra ensemble member is included solely because it performed well alone. The final two-member fusion fits well below the asset limit. Archival CatBoost, sparse-text and later model searches are evidence of explored alternatives; their weights are not required to reproduce this release's predictions.

<!-- pagebreak -->

## 5. Failure analysis, resources and reproduction

**Semantic modeling can remain wrong despite correct arithmetic.** In validation question 44, three classes contain 46 males and 48 females. The correct number without an opposite-gender partner is 2. The chosen candidate computes 94-46=48: its subtraction is numerically valid but counts pairs as individual students. The trace checker cannot repair a correctly evaluated expression with the wrong meaning. This illustrates why arithmetic consistency is weaker than semantic verification.

**Trace coverage is incomplete.** In validation question 52, a dog at 24 mph catches a rabbit at 15 mph with a 0.6-mile lead. The correct time is (0.6/9)*60=4 minutes. Several candidates claim 0.6/9=2/15 and answer 8. The final model still selects 8: the supplied LaTeX/unit formatting is not fully captured by the restricted equality parser. A synthetic plain expression can be corrected while the original formatted trace is missed. The four recorded validation failures in evaluation/validation-failures.json make these limitations inspectable without test-case analysis.

| Resource measurement | Result and scope |
|---|---|
| Inference assets, uncompressed | 1,209,639 bytes = 1.1536 MiB, including weights, statistics, config and freeze record |
| Formal test prediction process | 0.6210 s, ARM Mac, Python 3.12.14 |
| Independent formal test scoring process | 0.0546 s; total 0.6756 s |
| Earlier proxy validation, Xeon Gold 6248 | 3.3516 s, original source layout; not reference-CPU certification |
| Clean final-recipe GPU replay | {{retraining_seconds}} s, 12 fits, RTX 3080 Ti 12 GB |
| Recorded cumulative earlier search budget | 5,683.66 s (1.579 h); partial accounting, excluding unlogged work/waiting |

Inference uses NumPy and threadpoolctl. Sequential fresh-process timing includes startup, imports, loading, features, all members and independent scoring, with four library threads. The saved local CPU identifier is only ARM. Xeon Platinum 8457C compliance remains unverified. RTX 3080 Ti training is below the permitted RTX 4090 ceiling.

The clean GPU replay used Python 3.12.3, PyTorch 2.8.0+cu128, NumPy 2.3.2 and scikit-learn 1.7.1: **{{retraining_status}}** (verification/retraining.json). README.md gives installation, pinned data-fetch, training and evaluation commands. Direct checkpoint evaluation requires no retraining. FREEZE.json binds inference source and assets; RELEASE.json identifies packaged contents, with the student's published Git commit supplying the final code version.

## 6. Limitations and conclusion

Supervised verification and structure account for most gains; the arithmetic residual adds only two validation successes. Semantic errors, formatting gaps and incorrect consensus persist. Adaptive development on a small bank, no unseen audit evaluation and repeated test feedback prevent a clean generalization claim and may affect eligibility. Exact checkpoint reproduction does not remove these limitations.

OpenAI Codex substantially assisted design, coding, experiments, analysis, writing and packaging. Reused official evaluator/parser code, GSM8K attribution and AI assistance are documented in AI_REUSE.md and THIRD_PARTY.md. No additional pretrained weights or external training examples were introduced. Publication and course issue submission are performed by the student.

## References

[1] Karl Cobbe et al. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021. https://arxiv.org/abs/2110.14168

[2] DASE7506 MP2: Selecting Reliable LLM Reasoning, official course guide and bank. Pinned repository revision bb801f4f2e7629d270df765ab12c333b2dc616bb. https://github.com/xudongwu-0/DASE7506-MP2
