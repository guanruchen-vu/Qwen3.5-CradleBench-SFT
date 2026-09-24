#!/usr/bin/env python3
"""Export a run's end-of-epoch checkpoint as <run>/epoch<k>-adapter, laid out like best-adapter.

Usage: export_epoch_adapter.py RUN [EPOCH]  (EPOCH defaults to the final epoch).

The processor files and training-config.json come from best-adapter; the adapter weights come
from the checkpoint saved at the end of the last epoch. Idempotent: an existing export is
verified against its checkpoint instead of being rewritten.
"""

import filecmp
import json
from pathlib import Path
import shutil
import sys


def main():
    run = Path(sys.argv[1])
    config = json.loads((run / "resolved-config.json").read_text())
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else config["num_train_epochs"]
    if not 1 <= epochs <= config["num_train_epochs"]:
        raise SystemExit(f"epoch must be between 1 and {config['num_train_epochs']}")
    matches = []
    for checkpoint in sorted(run.glob("checkpoint-*")):
        state_path = checkpoint / "checkpoint-state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            if state["epoch"] == epochs and state["next_group"] == 0:
                matches.append((checkpoint, state))
    if len(matches) != 1:
        raise SystemExit(f"{run}: expected one end-of-epoch-{epochs} checkpoint, found {[m[0].name for m in matches]}")
    checkpoint, state = matches[0]
    output = run / f"epoch{epochs}-adapter"
    weights = ("adapter_config.json", "adapter_model.safetensors")
    if output.exists():
        if not all(filecmp.cmp(checkpoint / name, output / name, shallow=False) for name in weights):
            raise SystemExit(f"{output} exists but does not match {checkpoint}")
        print(f"verified {output} = {checkpoint.name}")
        return
    temporary = run / f".epoch{epochs}-adapter.incomplete"
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.copytree(run / "best-adapter", temporary)
    for name in weights:
        shutil.copy2(checkpoint / name, temporary / name)
    selection = {"note": f"end of epoch {epochs}; not necessarily the checkpoint selected by crisis Macro F1",
                 "checkpoint": checkpoint.name, **state}
    (temporary / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    temporary.rename(output)
    print(f"exported {output} from {checkpoint.name}")


if __name__ == "__main__":
    main()
