# Reproducibility and submission instructions - DASE7506 MP2

Student ID: **3036802038**. The release selects one of four supplied solutions using a compact linear/MLP verifier and bounded arithmetic-trace evidence. The included checkpoint runs directly on CPU; retraining is optional.

**Eligibility limitation:** development included repeated aggregate test-reference feedback and a test-informed stopping objective. This conflicts with the course test-use restriction. The final score is exploratory, not an unbiased untouched-test estimate. Packaging cannot repair that history; instructor review is required. Read [DISCLOSURE.md](../DISCLOSURE.md) before interpreting or submitting the result. The reference Xeon CPU has not been measured.

| Frozen release | Result |
|---|---:|
| Official validation | 743 / 798 = 93.10776942355889% |
| Validation disagreement subset | 300 / 355 = 84.50704225352112% |
| Final official test | 394 / 467 = **84.36830835117773%** |
| Total uncompressed inference assets | 1,209,639 bytes = 1.1536 MiB |
| Fresh test prediction + independent scoring, local ARM Mac | 0.6756 s; four-thread cap |

Only the final formal numerical test result is retained. No test predictions, test-case analysis or earlier numerical test scores are in this repository. All original benchmark files are fetched separately and hash-verified. This does not mean the earlier test queries did not occur.

## Install and reproduce the checkpoint score

Use Python **3.12**. The clean inference installation requires only NumPy and threadpoolctl.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/fetch_data.py
python code/check_data.py
python code/evaluate.py --split validation --output-dir runs/checkpoint-validation
python code/evaluate.py --split test --output-dir runs/checkpoint-test
```

Run commands from the repository root. Data download needs network access once, before evaluation. Offline copying is supported: `python code/fetch_data.py --source-dir /path/to/official/code/data`. Download is pinned to upstream commit `bb801f4f2e7629d270df765ab12c333b2dc616bb`; each file must match the supplied manifest. The benchmark is not changed or regenerated. Once installed, inference and scoring are offline and require no GPU, API, pretrained model or retraining.

`evaluate.py` verifies the checkpoint/source hashes and asset cap, then externally times two fresh, sequential processes running the unchanged official prediction and scoring scripts. It caps BLAS/OpenMP threads at four and writes the official `*-score.json` plus a separate resource record. Test choices exist only in a temporary file removed after scoring. The included `evaluation/test-score.json` is the final formal result; reproduction should use a fresh output directory as above. Timing varies by hardware and does not establish the reference-machine limit.

The public interface remains `predict(bank, arrays, checkpoint) -> list[int]`, in bank order. It accepts varying bank sizes and IDs. Supply `code/submission.py` and **this release's** `checkpoint/` together. Do not pair the method with an older research-workspace checkpoint.

For the exact underlying official commands (equivalent to the wrapper, excluding external timing):

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 python code/predict.py --method code/submission.py --checkpoint checkpoint --split validation --output runs/validation-predictions.json
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 python code/score_predictions.py --split validation --predictions runs/validation-predictions.json --output runs/validation-score.json
```

## Retrain the released method

The recorded training environment is Linux, Python 3.12.3, RTX 3080 Ti 12 GB, PyTorch 2.8.0+cu128, NumPy 2.3.2, scikit-learn 1.7.1 and four CPU threads. Install the CUDA 12.8 PyTorch wheel for that environment:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r code/requirements-training.txt
python code/fetch_data.py --splits train validation
python code/train.py --device cuda --output runs/reproduction
python code/evaluate.py --checkpoint runs/reproduction/checkpoint --split validation --output-dir runs/retrained-validation
```

`--device cpu` is supported, but exact training arrays can vary with the device, library builds and floating-point reductions. Direct checkpoint evaluation is the authoritative score reproduction path. `verification/retraining.json` records the clean GPU replay and numerical comparison with the released arrays.

`configs/training.json` defines the final recipe and fixed fusion temperatures/weights inherited from original training-OOF calibration. `configs/folds.json` contains five fixed question-level folds. The command fits five fold models and one full model for each of the two members (12 fits), writes OOF predictions, and selects the residual correction from 50 train-OOF/validation combinations. Training helpers reject test-reference loading. All candidate labels matching the official parsed reference are positive. The final MLP uses 19 epochs, seed 42 (42+fold for folds), width 256, dropout 0.3, AdamW learning rate 0.001 and weight decay 0.01. The objective is BCE plus 0.5 times negative log correct probability mass. Learned normalizers use each fitting partition only. Full details are in the report.

This command replays the released recipe, not every exploratory search or undocumented historical test query. Earlier aggregate search/ablation records are retained for context with their original settings; those records do not certify a clean historical test boundary. The recorded compute total is a partial search accounting, not a complete wall-clock audit.

## Validation evidence and report

```bash
python code/inspect_validation.py
python code/reproduce_ablations.py --device cuda --output runs/ablations
python -m pip install -r code/requirements-report.txt
python report/build_report.py
```

The first command reproduces the baseline/final comparisons and essential parser, arithmetic and input-alignment checks without reading test references. The second optionally replays the historical controlled MLP/text ablations, including three MLP seeds. It is not required for scoring. CatBoost search records are archival and are not part of the released inference method or the optional ablation command. The [English PDF](../report/report.pdf) is at most ten pages; [editable source](../report/report.md), [Chinese explanation](../README.zh.md) and [claim/evidence review](../report/review.md) are included.

## Release contents

| Path | Purpose |
|---|---|
| `code/`, `configs/` | Standalone inference, final-recipe training, official evaluator and pinned requirements |
| `checkpoint/` | Two members, normalizers, fusion config and `FREEZE.json` |
| `evaluation/` | Official final evaluation JSON, CPU records and validation-only analysis |
| `experiments/` | Training/validation search, controlled ablations, OOF/residual records |
| `verification/` | Interface and clean-retraining evidence, asset inventory |
| `provenance/` | Upstream revision/hashes, course guide and historical incident |
| `RELEASE.json` | Content-based source identity and file hashes, independent of future Git publication |
| `SUBMISSION.json` | Score/student ID and publication fields to fill manually |

Original scorer/parser files are byte-identical to the pinned upstream version. `checkpoint/FREEZE.json` binds the inference source to the assets. `RELEASE.json` pins the packaged source and evidence; a Git commit must additionally identify the version published by the student. Attribution and substantive AI assistance are disclosed in [AI_REUSE.md](../AI_REUSE.md) and [THIRD_PARTY.md](../THIRD_PARTY.md). Historical notes may describe earlier states; the current [disclosure](../DISCLOSURE.md) takes precedence, including over earlier claims about test-feedback exclusion.

## Publish and submit yourself

This workspace directory is an independent local Git repository. The ZIP extracts to `DASE7506_MP2/`. ZIP distributions omit Git internals; initialize Git after extracting the ZIP. No remote repository, push or course issue has been created. Configure your own Git identity and create the initial commit:

```bash
git init -b main
git config user.name "YOUR NAME"
git config user.email "YOUR EMAIL"
git add .
git commit -m "Release MP2 candidate verifier and reproducibility materials"
git rev-parse HEAD
```

Create a public GitHub repository and push this commit yourself. Upload the separately provided checkpoint ZIP to a GitHub Release, or use an immutable public checkpoint link. Record the full code commit and checkpoint link locally (this updates the form draft, not the frozen model):

```bash
python code/prepare_submission.py --code-url https://github.com/YOU/REPO/tree/FULL_COMMIT_SHA --checkpoint-url https://github.com/YOU/REPO/releases/download/TAG/checkpoint_submission_3036802038.zip --revision FULL_COMMIT_SHA
```

Use student ID **3036802038**, score **84.36830835117773**, the immutable code URL and checkpoint URL on the [course submission page](https://xudongwu-0.github.io/courses/dase7506/mp2/). The website only prepares a link: click **Submit on GitHub** and create the issue yourself, then verify that it exists. Disclose the test-use limitation to the instructor when asking about eligibility. Keep the published code, checkpoint and data-fetch path available for at least seven days after the deadline (31 October 2026, 23:59 UTC+8). `SUBMISSION.json` remains a local draft until publication fields are filled; editing those fields after the release does not change the scored checkpoint.

## Package the release

Run `python scripts/release.py verify` followed by `python scripts/release.py package` from the repository root. Both ZIPs and SHA256SUMS.json are written to `dist/`, which is ignored by Git. The repository ZIP uses `DASE7506_MP2/` as its top-level directory.
