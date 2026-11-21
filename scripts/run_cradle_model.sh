#!/usr/bin/env bash
# One model on CRADLE Bench, one L4 GPU, in order: BF16 zero-shot test, then Unanimous,
# Unanimous-IPW, and Consensus LoRA SFT. Each run's end-of-epoch-3 adapter is scored on the
# 600 test posts with the same decoding as the Qwen3.5 runs (greedy, --max-input-tokens 4608).
#
# Usage (from anywhere):
#   setsid nohup scripts/run_cradle_model.sh gemma4_e4b gemma4-e4b google/gemma-4-E4B-it \
#       ee0ef6023621cff504d758262d4e04895a5af4a2 4 > outputs/logs/run-gemma4-e4b.log 2>&1 &
# Arguments: <config prefix> <output prefix> <model> <revision> <test batch size>
# Re-running skips finished steps; an interrupted training run must be resumed by hand
# (train_lora_sft.py --resume-from-checkpoint <latest checkpoint>).
set -euo pipefail
cd "$(dirname "$0")/.."
cfg_prefix=$1 out_prefix=$2 model=$3 revision=$4 batch=${5:-4}
PY=${PY:-$HOME/.venvs/qwen/bin/python}
mkdir -p outputs/logs

score() {  # score <predictions.jsonl> <metrics.json> <log name>
  $PY scripts/analyze_baseline.py "$1" --expected-rows 600 --output "$2" > "outputs/logs/metrics-$3.txt"
}

echo "$(date -u +%FT%TZ) ${out_prefix} zero-shot"
base=outputs/${out_prefix}-bf16-baseline-test.jsonl
if [[ ! -f $base ]]; then
  $PY scripts/run_baseline.py --model "$model" --revision "$revision" --split test \
    --batch-size "$batch" --max-input-tokens 4608 --output "$base.partial" \
    > "outputs/logs/${out_prefix}-test-baseline.log" 2>&1
  mv "$base.partial" "$base"
fi
score "$base" "outputs/${out_prefix}-bf16-baseline-test-metrics.json" "${out_prefix}-baseline"

for run in unanimous unanimous_ipw consensus; do
  case $run in
    consensus) suffix="" ;;
    *) suffix="_$run" ;;
  esac
  out=outputs/${out_prefix}-lora-sft-${run//_/-}-seed42
  echo "$(date -u +%FT%TZ) ${out_prefix} train ${run}"
  if [[ ! -d $out/best-adapter ]]; then
    $PY scripts/train_lora_sft.py --mode train --config "configs/${cfg_prefix}_lora_sft${suffix}.json" \
      --output-dir "$out" > "outputs/logs/${out_prefix}-train-${run}.log" 2>&1
  fi
  $PY scripts/export_epoch_adapter.py "$out" 3
  echo "$(date -u +%FT%TZ) ${out_prefix} test ${run}"
  pred=outputs/${out_prefix}-lora-sft-${run//_/-}-epoch3-test.jsonl
  if [[ ! -f $pred ]]; then
    rm -f "$pred.partial"
    $PY scripts/run_sft_adapter.py --adapter "$out/epoch3-adapter" --split test --batch-size "$batch" \
      --max-input-tokens 4608 --output "$pred.partial" > "outputs/logs/${out_prefix}-test-${run}.log" 2>&1
    mv "$pred.partial" "$pred"
  fi
  score "$pred" "outputs/${out_prefix}-lora-sft-${run//_/-}-epoch3-test-metrics.json" "${out_prefix}-${run}"
done
echo "$(date -u +%FT%TZ) ${out_prefix} all runs finished"
