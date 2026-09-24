# Qwen3.5 × CRADLE Bench: Consensus vs. Unanimous training and label-IPW

This repository contains the **Qwen3.5-4B and Qwen3.5-9B** code used to train and evaluate CRADLE Bench classifiers on three training sets (LoRA SFT on the Consensus set, on the Unanimous set, and on the Unanimous set with a label-level inverse-probability-weighted loss), plus aggregate test-set results and analyses. This is a research/learning project, **not a clinical screening or crisis-response tool**.

## Model and data

- Base models: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) and [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`), publicly released post-trained Qwen3.5 models. The 9B configurations load a local `hf download --local-dir` copy (`model_path`) and check its download records against the pinned revision before loading.
- Dataset: [SungJoo/Cradle-Bench](https://huggingface.co/datasets/SungJoo/Cradle-Bench). The project uses two **alternative**, not combined, training sets: Consensus (4,181 posts) and Unanimous (3,058 posts), plus 420 development and 600 held-out test posts. The task predicts seven crisis types with ongoing/past timing, or `no_crisis` (15 labels total).
- The shared task prompt is in [`scripts/cradle_common.py`](scripts/cradle_common.py). Our concise prompt and comma-separated output format differ from the longer, JSON-based prompts in the [CRADLE Bench paper](https://aclanthology.org/2026.eacl-long.73.pdf); paper scores are therefore not a controlled reproduction of this setup.

## Training and evaluation

All six runs (two model sizes × three training sets) use **unquantized BF16 LoRA SFT** with identical settings: rank 16, alpha 32, dropout 0.05, learning rate `1e-4`, one post per microbatch, 16 gradient-accumulation steps, sequence limit 2,048, cosine schedule with 3% warmup, three epochs, seed 42. Training optimizes only answer tokens. **Every reported SFT result uses the checkpoint at the end of epoch 3; no checkpoint is selected on validation data, so the main results use only the training and test sets.** The trainer still scores the 420 development posts after each epoch; those numbers appear in the training summaries but do not pick checkpoints. The BF16 baselines use the same prompt and deterministic decoding, without an adapter. The 9B runs fit on a single 24 GB NVIDIA L4 (peak 20.3 GiB reserved).

**Unanimous-IPW.** Every Unanimous post is also a Consensus post, but the posts that were dropped are not missing at random: rare labels lose up to 61% of their posts (`childabuse_endangerment_ongoing` keeps 15 of 39) while `no_crisis` keeps 88%. The IPW runs give each Unanimous training post the weight *Consensus count / Unanimous count* of its labels (the largest one for multi-label posts), normalized to mean 1 over the training set (range 0.80–1.83; the `[0.5, 5]` clip is inactive). Only this per-post loss weight differs from the Unanimous runs: seed, data order, and LoRA initialization are the same. The weights use Consensus label counts only, never its posts. See `loss_weighting` in the `*_unanimous_ipw.json` configs and `label_ipw_weights` in [`scripts/sft_training_utils.py`](scripts/sft_training_utils.py).

**Evaluation.** Test inference uses greedy decoding and `--max-input-tokens 4608`, so no test post is truncated (the longest test prompt is 4,102 tokens). The batch size is 4 for 4B and 1 for 9B, which keeps 9B inference within 24 GB. Training-time validation keeps the configs' 2,048-token limit (the longest validation prompt is 2,027 tokens). Earlier versions of these results truncated six test posts at 2,048 tokens and reported validation-selected checkpoints; both are superseded.

Core entry points, in run order:

| Purpose | File |
| --- | --- |
| Download and validate the four official CSV files | [`scripts/download_cradle.py`](scripts/download_cradle.py) |
| Run the unquantized BF16 base model | [`scripts/run_baseline.py`](scripts/run_baseline.py) |
| Train a LoRA adapter | [`scripts/train_lora_sft.py`](scripts/train_lora_sft.py) with [`configs/`](configs/) |
| Export the end-of-epoch-3 checkpoint | [`scripts/export_epoch_adapter.py`](scripts/export_epoch_adapter.py) |
| Evaluate an exported adapter | [`scripts/run_sft_adapter.py`](scripts/run_sft_adapter.py) |
| Score predictions | [`scripts/analyze_baseline.py`](scripts/analyze_baseline.py) |
| Compare models and plot training | [`scripts/compare_metrics.py`](scripts/compare_metrics.py), [`scripts/plot_sft_results.py`](scripts/plot_sft_results.py) |
| Calculate paired bootstrap intervals | [`scripts/bootstrap_ci_summary.py`](scripts/bootstrap_ci_summary.py) |
| Relate label retention to per-label test recall | [`scripts/ipw_label_analysis.py`](scripts/ipw_label_analysis.py) |

The published `train_lora_sft.py` imports `lora_model_setup.py`, which holds the shared BF16/LoRA initialization and JSON helpers, so the formal trainer does not depend on a smoke-test file. To rerun, install PyTorch, a Transformers release supporting Qwen3.5, PEFT, Accelerate, `huggingface_hub`, NumPy, and Matplotlib; download the models locally before training because the SFT loader uses `local_files_only=True`. Run commands and options are documented in each script's `--help` or module docstring. The training configurations are Consensus, Unanimous, and Unanimous-IPW for [4B](configs/qwen35_4b_lora_sft.json) ([Unanimous](configs/qwen35_4b_lora_sft_unanimous.json), [IPW](configs/qwen35_4b_lora_sft_unanimous_ipw.json)) and [9B](configs/qwen35_9b_lora_sft.json) ([Unanimous](configs/qwen35_9b_lora_sft_unanimous.json), [IPW](configs/qwen35_9b_lora_sft_unanimous_ipw.json)). For example:

```bash
python scripts/train_lora_sft.py --mode train --config configs/qwen35_4b_lora_sft_unanimous_ipw.json --output-dir outputs/qwen35-4b-lora-sft-unanimous-ipw-seed42
python scripts/export_epoch_adapter.py outputs/qwen35-4b-lora-sft-unanimous-ipw-seed42 3
python scripts/run_sft_adapter.py --adapter outputs/qwen35-4b-lora-sft-unanimous-ipw-seed42/epoch3-adapter --split test --batch-size 4 --max-input-tokens 4608 --output outputs/qwen35-4b-lora-sft-unanimous-ipw-epoch3-test.jsonl
```

## Held-out test results (epoch 3)

All nine metrics below are point estimates with **95% percentile bootstrap CIs** from 10,000 paired resamples of the same 600 test posts (seed 42). Values are percentages: point estimate [CI lower, CI upper].

### Qwen3.5-4B

| Metric | Qwen3.5-4B BF16 | Qwen3.5-4B Consensus SFT Epoch3 | Qwen3.5-4B Unanimous SFT Epoch3 | Qwen3.5-4B Unanimous-IPW SFT Epoch3 |
| --- | ---: | ---: | ---: | ---: |
| Exact Match | 46.50% [42.50%, 50.50%] | 75.83% [72.50%, 79.17%] | 75.00% [71.50%, 78.33%] | 76.00% [72.67%, 79.33%] |
| Sample Jaccard | 49.21% [45.28%, 53.14%] | 80.07% [77.06%, 82.99%] | 78.39% [75.31%, 81.46%] | 79.42% [76.39%, 82.42%] |
| Micro F1 | 48.66% [44.77%, 52.52%] | 79.74% [76.71%, 82.72%] | 77.97% [74.82%, 81.09%] | 78.88% [75.70%, 81.95%] |
| Macro F1 · all 15 | 36.96% [32.49%, 40.95%] | 72.08% [65.88%, 76.54%] | 68.18% [61.99%, 73.32%] | 70.16% [64.05%, 74.89%] |
| Macro F1 · crisis 14 | 35.25% [30.50%, 39.43%] | 71.11% [64.56%, 75.87%] | 67.07% [60.48%, 72.47%] | 69.12% [62.66%, 74.13%] |
| Micro Recall | 47.46% [43.56%, 51.42%] | 81.04% [78.05%, 84.04%] | 78.96% [75.78%, 82.11%] | 79.70% [76.51%, 82.81%] |
| Macro Recall · all 15 | 43.28% [38.65%, 48.65%] | 74.69% [68.53%, 80.51%] | 70.11% [64.84%, 76.02%] | 72.97% [67.15%, 78.76%] |
| Macro Recall · crisis 14 | 42.60% [37.68%, 48.25%] | 74.34% [67.78%, 80.58%] | 69.55% [63.93%, 75.91%] | 72.65% [66.44%, 78.84%] |
| Valid Format | 96.50% [95.00%, 98.00%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] |

### Qwen3.5-9B

| Metric | Qwen3.5-9B BF16 | Qwen3.5-9B Consensus SFT Epoch3 | Qwen3.5-9B Unanimous SFT Epoch3 | Qwen3.5-9B Unanimous-IPW SFT Epoch3 |
| --- | ---: | ---: | ---: | ---: |
| Exact Match | 43.33% [39.33%, 47.33%] | 79.17% [76.00%, 82.33%] | 78.83% [75.67%, 82.00%] | 79.00% [75.67%, 82.17%] |
| Sample Jaccard | 46.67% [42.87%, 50.52%] | 82.41% [79.53%, 85.25%] | 82.44% [79.60%, 85.25%] | 82.59% [79.76%, 85.37%] |
| Micro F1 | 48.01% [44.27%, 51.75%] | 81.43% [78.32%, 84.45%] | 81.75% [78.73%, 84.69%] | 82.25% [79.37%, 85.08%] |
| Macro F1 · all 15 | 34.96% [30.25%, 38.86%] | 74.70% [68.54%, 79.36%] | 72.48% [66.58%, 77.12%] | 75.56% [69.57%, 79.93%] |
| Macro F1 · crisis 14 | 32.20% [27.18%, 36.33%] | 73.81% [67.28%, 78.74%] | 71.49% [65.20%, 76.41%] | 74.81% [68.45%, 79.45%] |
| Micro Recall | 46.87% [43.13%, 50.60%] | 83.13% [80.18%, 86.06%] | 82.24% [79.19%, 85.19%] | 82.99% [80.12%, 85.80%] |
| Macro Recall · all 15 | 38.01% [33.99%, 42.32%] | 79.88% [73.69%, 85.08%] | 74.98% [69.23%, 80.71%] | 77.93% [72.18%, 83.47%] |
| Macro Recall · crisis 14 | 36.04% [31.78%, 40.61%] | 79.71% [73.07%, 85.29%] | 74.61% [68.47%, 80.72%] | 77.82% [71.67%, 83.78%] |
| Valid Format | 93.17% [91.17%, 95.17%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] |

At 4B, Unanimous − Consensus is -4.05 pp [-9.97, +1.79] in crisis Macro F1 and -4.79 pp [-11.04, +1.07] in crisis Macro Recall; Unanimous-IPW − Unanimous is +2.05 pp [-2.61, +7.38] and +3.10 pp [-1.73, +8.86]. Over the 14 crisis labels, the Spearman correlation between label retention and the Unanimous − Consensus recall difference is ρ = +0.209, permutation p = 0.473. At 9B, Unanimous − Consensus is -2.32 pp [-7.27, +2.30] in crisis Macro F1 and -5.10 pp [-10.84, -0.10] in crisis Macro Recall; Unanimous-IPW − Unanimous is +3.32 pp [-0.05, +6.97] and +3.21 pp [+0.01, +6.77]. Over the 14 crisis labels, the Spearman correlation between label retention and the Unanimous − Consensus recall difference is ρ = +0.567, permutation p = 0.038. Exact Match and Micro F1 differ by at most 1.8 pp between any two SFT models of the same size. These are single-seed results; the CIs describe sampling uncertainty on this test set, not training variance, and are neither significance tests nor clinical validation.

Result files, for [4B](outputs/qwen35-4b) and [9B](outputs/qwen35-9b):

- `test-bootstrap-ci/`: test summary with all paired differences (`summary.md`), machine-readable summary (`summary.csv`), and paired differences (`paired-differences.csv`).
- `ipw-label-analysis/`: label retention, IPW weights, and per-label test recall, with Spearman correlations between retention and recall differences.
- `training/{consensus,unanimous,unanimous-ipw}/`: validation metrics after each epoch and the training-loss curve. Runs that skipped their own before-training evaluation reuse the result of a run with the same untrained model.

Macro F1 follows the analysis script's convention of assigning F1 = 0 to a label absent from a resample. Raw data and individual predictions are not committed. Download the dataset from its original source and run the scripts to regenerate them.
