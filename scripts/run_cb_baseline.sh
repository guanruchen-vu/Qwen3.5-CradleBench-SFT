#!/usr/bin/env bash
# Class-balanced baseline on the Unanimous set (loss_weighting.method = label_cb, beta 0.99, the
# same clip as IPW), one L4 GPU, models in the order given. For each model: train, export the
# end-of-epoch-3 adapter, and score it on the 600 test posts with the same decoding as the other
# runs (greedy, --max-input-tokens 4608, the model's usual test batch size).
#
# Usage (from anywhere):
#   setsid nohup scripts/run_cb_baseline.sh qwen35_4b:qwen35-4b:4 qwen35_9b:qwen35-9b:1 \
#       > outputs/logs/run-cb-qwen.log 2>&1 &
# Each argument: <config prefix>:<output prefix>:<test batch size>
# Re-running skips finished steps; an interrupted training run must be resumed by hand
# (train_lora_sft.py --resume-from-checkpoint <latest checkpoint>).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/qwen/bin/python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p outputs/logs

for spec in "$@"; do
  IFS=: read -r cfg_prefix out_prefix batch <<< "$spec"
  out=outputs/${out_prefix}-lora-sft-unanimous-cb-seed42
  echo "$(date -u +%FT%TZ) ${out_prefix} train unanimous_cb"
  if [[ ! -d $out/best-adapter ]]; then
    $PY scripts/train_lora_sft.py --mode train --config "configs/${cfg_prefix}_lora_sft_unanimous_cb.json" \
      --output-dir "$out" > "outputs/logs/${out_prefix}-train-unanimous_cb.log" 2>&1
  fi
  $PY scripts/export_epoch_adapter.py "$out" 3
  echo "$(date -u +%FT%TZ) ${out_prefix} test unanimous_cb"
  pred=outputs/${out_prefix}-lora-sft-unanimous-cb-epoch3-test.jsonl
  if [[ ! -f $pred ]]; then
    rm -f "$pred.partial"
    $PY scripts/run_sft_adapter.py --adapter "$out/epoch3-adapter" --split test --batch-size "$batch" \
      --max-input-tokens 4608 --output "$pred.partial" > "outputs/logs/${out_prefix}-test-unanimous_cb.log" 2>&1
    mv "$pred.partial" "$pred"
  fi
  $PY scripts/analyze_baseline.py "$pred" --expected-rows 600 \
    --output "outputs/${out_prefix}-lora-sft-unanimous-cb-epoch3-test-metrics.json" \
    > "outputs/logs/metrics-${out_prefix}-unanimous_cb.txt"
done
echo "$(date -u +%FT%TZ) all class-balanced runs finished"
