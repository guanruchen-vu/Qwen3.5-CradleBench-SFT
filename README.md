# Qwen3.5 × CRADLE Bench: BF16 baseline and LoRA SFT

This repository contains the **Qwen3.5-4B** code used to train and evaluate two CRADLE Bench classifiers, plus aggregate test-set results. It excludes the smoke test, the 9B experiment, raw Reddit posts, per-post predictions, model weights, checkpoints, and machine-specific credentials. This is a research/learning project, **not a clinical screening or crisis-response tool**.

## Model and data

- Base model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), a publicly released post-trained Qwen3.5 model. The SFT configurations pin revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Dataset: [SungJoo/Cradle-Bench](https://huggingface.co/datasets/SungJoo/Cradle-Bench). The project uses two **alternative**, not combined, training sets: Consensus (4,181 posts) and Unanimous (3,058 posts), plus 420 development and 600 held-out test posts. The task predicts seven crisis types with ongoing/past timing, or `no_crisis` (15 labels total).
- The shared task prompt is in [`scripts/cradle_common.py`](scripts/cradle_common.py). Our concise prompt and comma-separated output format differ from the longer, JSON-based prompts in the [CRADLE Bench paper](https://aclanthology.org/2026.eacl-long.73.pdf); paper scores are therefore not a controlled reproduction of this setup.

## Training and evaluation

Both runs use **unquantized BF16 LoRA SFT** on Qwen3.5-4B: rank 16, alpha 32, dropout 0.05, learning rate `1e-4`, one post per microbatch, 16 gradient-accumulation steps, sequence limit 2,048, cosine schedule with 3% warmup, and up to three epochs (seed 42). The SFT trainer selects checkpoints using validation crisis-only Macro F1: **Consensus epoch 3** and **Unanimous epoch 1** were used for the test comparison. The BF16 baseline uses the same prompt and deterministic decoding, without an adapter. Training optimizes only answer tokens.

Core entry points, in run order:

| Purpose | File |
| --- | --- |
| Download and validate the four official CSV files | [`scripts/download_cradle.py`](scripts/download_cradle.py) |
| Run the unquantized BF16 base model | [`scripts/run_baseline.py`](scripts/run_baseline.py) |
| Train either LoRA adapter | [`scripts/train_lora_sft.py`](scripts/train_lora_sft.py) with [`configs/`](configs/) |
| Evaluate an exported adapter | [`scripts/run_sft_adapter.py`](scripts/run_sft_adapter.py) |
| Score predictions | [`scripts/analyze_baseline.py`](scripts/analyze_baseline.py) |
| Compare models and plot training | [`scripts/compare_metrics.py`](scripts/compare_metrics.py), [`scripts/plot_sft_results.py`](scripts/plot_sft_results.py) |
| Calculate paired bootstrap intervals | [`scripts/bootstrap_ci_summary.py`](scripts/bootstrap_ci_summary.py) |

The published `train_lora_sft.py` imports `lora_model_setup.py`: this is a verbatim extraction of its two shared BF16/LoRA initialization and JSON helpers, so the formal trainer does not depend on a smoke-test file. To rerun, install PyTorch, a Transformers release supporting Qwen3.5, PEFT, Accelerate, `huggingface_hub`, NumPy, and Matplotlib; download the model locally before training because the SFT loader uses `local_files_only=True`. Run commands and options are documented in each script's `--help` or module docstring. The training configurations are [`Consensus`](configs/qwen35_4b_lora_sft.json) and [`Unanimous`](configs/qwen35_4b_lora_sft_unanimous.json).

## Held-out test results

Point estimates and **95% percentile bootstrap CIs** from 10,000 paired resamples of the same 600 test posts (seed 42):

| Model | Exact Match | Micro F1 | Crisis-only Macro F1 |
| --- | ---: | ---: | ---: |
| BF16 base | 45.83% [41.83%, 49.83%] | 48.31% [44.41%, 52.18%] | 34.77% [30.05%, 38.98%] |
| Consensus SFT, epoch 3 | 75.00% [71.50%, 78.50%] | 79.47% [76.44%, 82.41%] | 70.89% [64.29%, 75.59%] |
| Unanimous SFT, epoch 1 | 74.50% [71.00%, 78.00%] | 77.27% [73.91%, 80.49%] | 63.53% [57.32%, 68.32%] |

The **three published CI result files** are the [readable summary](outputs/bf16base-vs-sft-all-test/bootstrap-ci/summary.md), [machine-readable summary table](outputs/bf16base-vs-sft-all-test/bootstrap-ci/summary.csv), and [paired-differences table](outputs/bf16base-vs-sft-all-test/bootstrap-ci/paired-differences.csv). Paired differences subtract the earlier-listed model from the later-listed model and report percentage points. The CIs describe sampling uncertainty on this test set; they are not clinical validation.

Raw data and individual predictions are intentionally not committed. Download the dataset from its original source and run the scripts to regenerate them.
