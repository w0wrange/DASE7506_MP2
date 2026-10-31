# Project 2 — Selecting Reliable LLM Reasoning

`DASE7506_MP2/` is the submission repository for Project 2. Student ID: **3036802038**. The final method combines logistic regression on fixed embeddings, a small neural ranker and arithmetic trace correction to select one of four candidate answers per question.

## Repository structure

- `code/`: prediction interface, training entry point, official data checker and independent scoring scripts.
- `configs/`: training configuration and five fixed question-level folds.
- `checkpoint/`: final weights, normalization statistics, fusion configuration and freeze hashes.
- `evaluation/`: official evaluation JSON, resource measurements and validation analysis.
- `experiments/`, `verification/`: experiment, ablation and complete retraining verification records.
- `report/`: [five-page English report](report/report.pdf), editable source and build script.
- `provenance/COURSE_GUIDE.md`: assignment requirements; this directory also retains data provenance and original hashes.
- `requirements.txt`: pinned CPU inference dependencies. Training and report dependencies are listed separately in `code/requirements-*.txt`.
- [Detailed reproduction guide](docs/REPRODUCIBILITY.md): complete reproduction and manual submission instructions.

## Activate the environment

Use Python **3.12**. For a new installation, follow “Recreate the environment” below. If the environment already exists, run the following from the repository root:

```bash
source .venv/bin/activate
```

Evaluation runs on CPU with at most four computation threads. Evaluating the supplied checkpoint requires neither PyTorch nor retraining.

## Evaluate the frozen checkpoint

Install the official data, verify its hashes and evaluate the validation split:

```bash
python code/fetch_data.py
python code/check_data.py
python code/evaluate.py --split validation --output-dir runs/checkpoint-validation
```

Data installation is pinned to course repository revision `bb801f4f2e7629d270df765ab12c333b2dc616bb`. To copy an existing local dataset, add `--source-dir /path/to/official/code/data` to the fetch command. Prediction and scoring run offline after installation. Use a new output directory for each evaluation. The wrapper verifies the freeze hashes and measures the complete prediction process and independent scoring process.

The final formal test result is saved in `evaluation/test-score.json`. To reproduce it with the same checkpoint:

```bash
python code/evaluate.py --split test --output-dir runs/checkpoint-test
```

## Train the final model

The recorded training environment uses one RTX 3080 Ti, Python 3.12.3 and PyTorch 2.8.0+cu128. With the environment activated, install the training dependencies and run:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r code/requirements-training.txt
python code/fetch_data.py --splits train validation
python code/train.py --device cuda --output runs/reproduction
python code/evaluate.py --checkpoint runs/reproduction/checkpoint --split validation --output-dir runs/retrained-validation
```

Use a new output directory for each training run. The full-training seed is 42, and the five fold seeds are 42–46. The recipe performs 12 fits. Weights and normalization statistics are fitted only on training data; residual parameters are selected using training out-of-fold predictions and validation results. The complete configuration is in `configs/training.json`. CPU retraining is supported with `--device cpu`, although numerical results can vary across devices and library versions.

## Recreate the environment

Install Python 3.12, then run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Inference dependencies are NumPy 2.3.2 and threadpoolctl 3.6.0. To build the report:

```bash
python -m pip install -r code/requirements-report.txt
python report/build_report.py
```

## AI assistance and attribution

Codex substantially assisted research design, implementation, training orchestration, analysis, report writing and repository preparation. The official parser, data and independent scorer are unchanged. Assistance and reuse are discussed in the [report](report/report.pdf); the [data license](code/data/LICENSE.txt), [data attribution](code/data/NOTICE.txt) and [upstream provenance](provenance/upstream.json) are retained.

## Verification record (2026-10-31)

- Official validation: **743/798, 93.11%**; disagreement subset: **300/355, 84.51%**.
- Final formal test: **394/467, 84.36830835117773%**. Only this final formal test result and its resource record are retained.
- A clean inference environment passed official evaluation. Interface checks covered reordered questions, varying input sizes and changed IDs while preserving output alignment.
- Complete retraining on an RTX 3080 Ti took approximately **62.63 seconds**. All released weight and normalization arrays matched exactly, and validation accuracy was reproduced.
- Uncompressed inference assets total **1,209,639 bytes, approximately 1.154 MiB**.
- Complete test prediction plus independent scoring took **0.676 seconds** on the local ARM Mac. Runtime on the designated Xeon Platinum 8457C has not been measured.

## Experimental limitations and submission

Earlier exploration included repeated test-reference feedback and a test-score-based stopping objective, which conflicts with the course restrictions on test usage. The final result is therefore not an unbiased generalization estimate on an untouched test set; eligibility requires instructor review. This history is disclosed in the [report](report/report.pdf). Reorganizing the repository or retraining cannot remove this limitation.

The student publishes the GitHub repository and completes the course submission. Configure your own Git identity, create a commit, publish the code and matching checkpoint, and enter the public URLs and full commit hash in `SUBMISSION.json`. Then create the GitHub issue through the course website. Exact commands are in the [detailed reproduction guide](docs/REPRODUCIBILITY.md#publish-and-submit-yourself).

To generate release archives, run the following from the repository root. The first command regenerates the release manifest; the archives are written to the Git-ignored `dist/` directory:

```bash
python scripts/release.py manifest
python scripts/release.py verify
python scripts/release.py package
```
