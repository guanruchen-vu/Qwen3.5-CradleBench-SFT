"""Generate baseline-compatible predictions using a formal SFT best-adapter export."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from run_baseline import batched, generate_batch, load_rows, make_result
from sft_training_utils import validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True, help="The exported best-adapter directory")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((args.adapter / "training-config.json").read_text())
    validate_config(config)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 adapter inference requires a CUDA GPU")
    set_seed(config["seed"])
    processor = AutoProcessor.from_pretrained(args.adapter, local_files_only=True)
    base = AutoModelForMultimodalLM.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    options = SimpleNamespace(text=None, index=None, limit=None, split=args.split,
                              data_dir=Path(config["data_dir"]),
                              max_input_tokens=config["eval_max_input_tokens"], max_new_tokens=config["max_new_tokens"])
    rows = load_rows(options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        count = 0
        for batch in batched(rows, args.batch_size):
            responses = generate_batch(torch, processor, model, batch, options)
            for row, response in zip(batch, responses, strict=True):
                destination.write(json.dumps(make_result(row, response, args.split), ensure_ascii=False) + "\n")
            destination.flush()
            count += len(batch)
            print(f"Completed {count}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
