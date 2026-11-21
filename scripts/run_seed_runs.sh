#!/usr/bin/env bash
# Extra-seed runs, one L4 GPU, in the order given. For each <config stem>:<run name>:<test batch>
# train with configs/<config stem>.json, export the end-of-epoch-3 adapter, and score it on the
# 600 test posts with the same decoding as the seed-42 runs (greedy, --max-input-tokens 4608).
#
# Usage (from anywhere):
#   setsid nohup scripts/run_seed_runs.sh \
#       qwen35_4b_lora_sft_unanimous_seed67:qwen35-4b-lora-sft-unanimous-seed67:4 ... \
#       > outputs/logs/run-seeds-4b.log 2>&1 &
# Re-running skips finished steps; an interrupted training run must be resumed by hand
# (train_lora_sft.py --resume-from-checkpoint <latest checkpoint>).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/qwen/bin/python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p outputs/logs

for spec in "$@"; do
  IFS=: read -r cfg run batch <<< "$spec"
  out=outputs/$run
  echo "$(date -u +%FT%TZ) $run train"
  if [[ ! -d $out/best-adapter ]]; then
    $PY scripts/train_lora_sft.py --mode train --config "configs/$cfg.json" --output-dir "$out" \
      > "outputs/logs/$run-train.log" 2>&1
  fi
  $PY scripts/export_epoch_adapter.py "$out" 3
  echo "$(date -u +%FT%TZ) $run test"
  pred=outputs/$run-epoch3-test.jsonl
  if [[ ! -f $pred ]]; then
    rm -f "$pred.partial"
    $PY scripts/run_sft_adapter.py --adapter "$out/epoch3-adapter" --split test --batch-size "$batch" \
      --max-input-tokens 4608 --output "$pred.partial" > "outputs/logs/$run-test.log" 2>&1
    mv "$pred.partial" "$pred"
  fi
  $PY scripts/analyze_baseline.py "$pred" --expected-rows 600 \
    --output "outputs/$run-epoch3-test-metrics.json" > "outputs/logs/metrics-$run.txt"
done
echo "$(date -u +%FT%TZ) all seed runs finished"
