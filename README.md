# Qwen3.5 × CRADLE Bench: BF16 baseline, LoRA SFT, and label-IPW

This repository contains the **Qwen3.5-4B** code used to train and evaluate three CRADLE Bench classifiers (LoRA SFT on the Consensus set, on the Unanimous set, and on the Unanimous set with a label-level inverse-probability-weighted loss), plus aggregate test-set results and analyses. This is a research/learning project, **not a clinical screening or crisis-response tool**.

## Model and data

- Base model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), a publicly released post-trained Qwen3.5 model. The SFT configurations pin revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Dataset: [SungJoo/Cradle-Bench](https://huggingface.co/datasets/SungJoo/Cradle-Bench). The project uses two **alternative**, not combined, training sets: Consensus (4,181 posts) and Unanimous (3,058 posts), plus 420 development and 600 held-out test posts. The task predicts seven crisis types with ongoing/past timing, or `no_crisis` (15 labels total).
- The shared task prompt is in [`scripts/cradle_common.py`](scripts/cradle_common.py). Our concise prompt and comma-separated output format differ from the longer, JSON-based prompts in the [CRADLE Bench paper](https://aclanthology.org/2026.eacl-long.73.pdf); paper scores are therefore not a controlled reproduction of this setup.

## Training and evaluation

All three runs use **unquantized BF16 LoRA SFT** on Qwen3.5-4B: rank 16, alpha 32, dropout 0.05, learning rate `1e-4`, one post per microbatch, 16 gradient-accumulation steps, sequence limit 2,048, cosine schedule with 3% warmup, and up to three epochs (seed 42). Training optimizes only answer tokens. The trainer selects checkpoints using validation crisis-only Macro F1: **Consensus epoch 3**, **Unanimous epoch 1**, and **Unanimous-IPW epoch 1** were used for the test comparison. The BF16 baseline uses the same prompt and deterministic decoding, without an adapter.

**Unanimous-IPW.** Every Unanimous post is also a Consensus post, but the posts that were dropped are not missing at random: rare labels lose up to 61% of their posts (`childabuse_endangerment_ongoing` keeps 15 of 39) while `no_crisis` keeps 88%. The IPW run gives each Unanimous training post the weight *Consensus count / Unanimous count* of its labels (the largest one for multi-label posts), normalized to mean 1 over the training set (range 0.80–1.83; the `[0.5, 5]` clip is inactive). Only this per-post loss weight differs from the Unanimous run: seed, data order, LoRA initialization, and checkpoint selection are the same. The weights use Consensus label counts only, never its posts. See `loss_weighting` in [`configs/qwen35_4b_lora_sft_unanimous_ipw.json`](configs/qwen35_4b_lora_sft_unanimous_ipw.json) and `label_ipw_weights` in [`scripts/sft_training_utils.py`](scripts/sft_training_utils.py).

**Evaluation.** Test inference uses greedy decoding, batch size 4, and `--max-input-tokens 4608`, so no test post is truncated (the longest test prompt is 4,102 tokens). An earlier version of these results used the 2,048-token training limit at test time, which truncated six test posts. Training-time validation keeps the configs' 2,048-token limit (the longest validation prompt is 2,027 tokens).

Core entry points, in run order:

| Purpose | File |
| --- | --- |
| Download and validate the four official CSV files | [`scripts/download_cradle.py`](scripts/download_cradle.py) |
| Run the unquantized BF16 base model | [`scripts/run_baseline.py`](scripts/run_baseline.py) |
| Train a LoRA adapter | [`scripts/train_lora_sft.py`](scripts/train_lora_sft.py) with [`configs/`](configs/) |
| Evaluate an exported adapter | [`scripts/run_sft_adapter.py`](scripts/run_sft_adapter.py) |
| Score predictions | [`scripts/analyze_baseline.py`](scripts/analyze_baseline.py) |
| Compare models and plot training | [`scripts/compare_metrics.py`](scripts/compare_metrics.py), [`scripts/plot_sft_results.py`](scripts/plot_sft_results.py) |
| Calculate paired bootstrap intervals | [`scripts/bootstrap_ci_summary.py`](scripts/bootstrap_ci_summary.py) |
| Relate label retention to per-label test recall | [`scripts/ipw_label_analysis.py`](scripts/ipw_label_analysis.py) |

The published `train_lora_sft.py` imports `lora_model_setup.py`, which holds the shared BF16/LoRA initialization and JSON helpers, so the formal trainer does not depend on a smoke-test file. To rerun, install PyTorch, a Transformers release supporting Qwen3.5, PEFT, Accelerate, `huggingface_hub`, NumPy, and Matplotlib; download the model locally before training because the SFT loader uses `local_files_only=True`. Run commands and options are documented in each script's `--help` or module docstring. The 4B training configurations are [`Consensus`](configs/qwen35_4b_lora_sft.json), [`Unanimous`](configs/qwen35_4b_lora_sft_unanimous.json), and [`Unanimous-IPW`](configs/qwen35_4b_lora_sft_unanimous_ipw.json). For example:

```bash
python scripts/train_lora_sft.py --mode train --config configs/qwen35_4b_lora_sft_unanimous_ipw.json --output-dir outputs/qwen35-4b-lora-sft-unanimous-ipw-seed42
python scripts/run_sft_adapter.py --adapter outputs/qwen35-4b-lora-sft-unanimous-ipw-seed42/best-adapter --split test --batch-size 4 --max-input-tokens 4608 --output outputs/qwen35-4b-lora-sft-unanimous-ipw-test.jsonl
```

## Held-out test results

All nine metrics below are point estimates with **95% percentile bootstrap CIs** from 10,000 paired resamples of the same 600 test posts (seed 42). Values are percentages: point estimate [CI lower, CI upper].

| Metric | Qwen3.5-4B BF16 | Qwen3.5-4B Consensus SFT Epoch3 | Qwen3.5-4B Unanimous SFT Epoch1 | Qwen3.5-4B Unanimous-IPW SFT Epoch1 |
| --- | ---: | ---: | ---: | ---: |
| Exact Match | 46.50% [42.50%, 50.50%] | 75.83% [72.50%, 79.17%] | 75.17% [71.83%, 78.50%] | 73.33% [69.83%, 76.83%] |
| Sample Jaccard | 49.21% [45.28%, 53.14%] | 80.07% [77.06%, 82.99%] | 78.26% [75.11%, 81.32%] | 76.83% [73.58%, 79.99%] |
| Micro F1 | 48.66% [44.77%, 52.52%] | 79.74% [76.71%, 82.72%] | 77.46% [74.11%, 80.68%] | 76.83% [73.64%, 79.97%] |
| Macro F1 · all 15 | 36.96% [32.49%, 40.95%] | 72.08% [65.88%, 76.54%] | 65.22% [59.42%, 69.75%] | 65.70% [59.25%, 70.78%] |
| Macro F1 · crisis 14 | 35.25% [30.50%, 39.43%] | 71.11% [64.56%, 75.87%] | 63.85% [57.71%, 68.66%] | 64.44% [57.58%, 69.85%] |
| Micro Recall | 47.46% [43.56%, 51.42%] | 81.04% [78.05%, 84.04%] | 75.67% [72.25%, 78.97%] | 75.22% [71.98%, 78.50%] |
| Macro Recall · all 15 | 43.28% [38.65%, 48.65%] | 74.69% [68.53%, 80.51%] | 62.95% [58.06%, 68.01%] | 64.06% [58.65%, 69.99%] |
| Macro Recall · crisis 14 | 42.60% [37.68%, 48.25%] | 74.34% [67.78%, 80.58%] | 61.57% [56.33%, 66.93%] | 63.26% [57.49%, 69.64%] |
| Valid Format | 96.50% [95.00%, 98.00%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] |

In paired differences, Unanimous − Consensus is −7.26 pp [−13.62, −0.33] in crisis Macro F1 and −12.77 pp [−19.61, −5.69] in crisis Macro Recall. IPW moves Unanimous by only +0.59 pp [−5.06, +6.08] and +1.70 pp [−3.47, +7.45], leaving Unanimous-IPW − Consensus at −6.67 pp [−13.43, −0.16] and −11.08 pp [−18.09, −4.61]. IPW does shift the label prior (`no_crisis` predictions: Unanimous 177, Unanimous-IPW 150, Consensus 160), but recall on the lowest-retention labels is not recovered. These are single-seed results; the CIs describe sampling uncertainty on this test set, not training variance, and are neither significance tests nor clinical validation.

Result files:

- [Test summary with all paired differences](outputs/qwen35-4b/test-bootstrap-ci/summary.md), [machine-readable summary](outputs/qwen35-4b/test-bootstrap-ci/summary.csv), and [paired differences](outputs/qwen35-4b/test-bootstrap-ci/paired-differences.csv).
- [Label retention, IPW weights, and per-label test recall](outputs/qwen35-4b/ipw-label-analysis/label-analysis.md) ([CSV](outputs/qwen35-4b/ipw-label-analysis/label-analysis.csv)), including Spearman correlations between retention and recall differences.
- Validation by epoch and training-loss curves for [Consensus](outputs/qwen35-4b/training/consensus/comparison.md), [Unanimous](outputs/qwen35-4b/training/unanimous/comparison.md), and [Unanimous-IPW](outputs/qwen35-4b/training/unanimous-ipw/comparison.md). The IPW run skipped its own before-training evaluation; its "Before" row reuses the Unanimous run's result for the same untrained model.

Macro F1 follows the analysis script's convention of assigning F1 = 0 to a label absent from a resample. Raw data and individual predictions are not committed. Download the dataset from its original source and run the scripts to regenerate them.
