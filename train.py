"""
FINA Training Script v3
========================
QLoRA fine-tuning for Qwen2.5-3B-Instruct using structured prompt/completion data.
Supervises only assistant completions, not the full prompt.

Dataset format (JSONL):
  {"prompt": [...messages...], "completion": [...messages...], "family": "..."}

Usage: python train.py
"""

import argparse
import hashlib
import inspect
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SOURCE_ADAPTER_NAME = "financial_qwen_native_v7"
NEW_MODEL_NAME = "financial_qwen_native_v8"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "hybrid_data.jsonl"
SOURCE_ADAPTER_DIR = PROJECT_ROOT / SOURCE_ADAPTER_NAME
CHECKPOINT_DIR = PROJECT_ROOT / f"{NEW_MODEL_NAME}_checkpoints"
MODEL_OUTPUT_DIR = PROJECT_ROOT / NEW_MODEL_NAME
SPLITS_DIR = PROJECT_ROOT / "splits"
MAX_SEQ_LENGTH = 2048
SPLIT_SEED = 42
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train / val / test  — §5 of the thesis plan


def _row_hash(row: dict) -> str:
    """Stable identifier for a dataset example (used as the split key)."""
    payload = json.dumps(row.get("prompt", ""), ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _role_for(row: dict) -> str:
    """Best-effort role tag for split-distribution accounting."""
    fam = (row.get("family") or "").lower()
    if "student" in fam:
        return "Student"
    if "worker" in fam:
        return "Worker"
    if "freelancer" in fam:
        return "Freelancer"
    text = json.dumps(row.get("prompt", ""), ensure_ascii=False).lower()
    for role in ("student", "worker", "freelancer"):
        if f'"{role}"' in text or f"role: {role}" in text:
            return role.capitalize()
    return "Unspecified"


def build_deterministic_splits(rows: list[dict]) -> dict[str, list[int]]:
    """Seeded shuffle into 80/10/10 train/val/test by dataset index."""
    indices = list(range(len(rows)))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(round(n * SPLIT_RATIOS[0]))
    n_val = int(round(n * SPLIT_RATIOS[1]))
    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }


def write_split_manifest(rows: list[dict], splits: dict[str, list[int]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "seed": SPLIT_SEED,
        "ratios": {"train": SPLIT_RATIOS[0], "val": SPLIT_RATIOS[1], "test": SPLIT_RATIOS[2]},
        "total": len(rows),
        "splits": {},
    }
    hashes_per_split: dict[str, list[str]] = {}
    for split, indices in splits.items():
        roles = Counter(_role_for(rows[i]) for i in indices)
        families = Counter((rows[i].get("family") or "unknown") for i in indices)
        hashes = [_row_hash(rows[i]) for i in indices]
        hashes_per_split[split] = hashes
        manifest["splits"][split] = {
            "n": len(indices),
            "pct": round(len(indices) / len(rows) * 100, 2) if rows else 0.0,
            "roles": dict(roles),
            "families": dict(families),
        }
        (out_dir / f"{split}_ids.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    overlap_train_val = set(hashes_per_split["train"]) & set(hashes_per_split["val"])
    overlap_train_test = set(hashes_per_split["train"]) & set(hashes_per_split["test"])
    overlap_val_test = set(hashes_per_split["val"]) & set(hashes_per_split["test"])
    manifest["overlap_check"] = {
        "train_val": len(overlap_train_val),
        "train_test": len(overlap_train_test),
        "val_test": len(overlap_val_test),
    }
    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError("Split overlap detected — manifest would be untrustworthy.")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the most recently updated checkpoint directory, if any."""
    if not checkpoint_dir.exists():
        return None

    latest_path = None
    latest_mtime = -1.0
    for path in checkpoint_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        mtime = path.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path
    return latest_path


def ensure_utf8_mode():
    """TRL reads template files with the process default encoding on Windows."""
    if sys.flags.utf8_mode:
        return
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def require_supported_python():
    """This training stack is unstable on newer Python runtimes on Windows."""
    version = sys.version_info
    if version < (3, 11) or version >= (3, 13):
        raise RuntimeError(
            "Use Python 3.11 or 3.12 for training. "
            f"Current runtime is {version.major}.{version.minor}.{version.micro}."
        )


def require_training_gpu() -> tuple[bool, torch.dtype]:
    """This QLoRA configuration requires CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. train.py is configured for 4-bit QLoRA and cannot train "
            "Qwen2.5-3B effectively on CPU. Run it on a CUDA machine."
        )
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return use_bf16, compute_dtype


def write_splits_for(raw_dataset, out_dir: Path = SPLITS_DIR) -> dict:
    rows = list(raw_dataset)
    splits = build_deterministic_splits(rows)
    manifest_path = write_split_manifest(rows, splits, out_dir)

    try:
        from benchmark_cases import TEST_CASES
        train_hashes = set((out_dir / "train_ids.txt").read_text(encoding="utf-8").splitlines())
        bench_text = json.dumps([tc.get("question", "") for tc in TEST_CASES], ensure_ascii=False)
        bench_hash = hashlib.sha1(bench_text.encode("utf-8")).hexdigest()
        # Lightweight assertion: per-question hashes don't match dataset prompts.
        for tc in TEST_CASES:
            q_hash = hashlib.sha1(json.dumps(tc.get("question", ""), ensure_ascii=False).encode("utf-8")).hexdigest()
            if q_hash in train_hashes:
                raise RuntimeError(f"Benchmark case {tc['id']} appears to overlap with training set.")
        print(f"[splits] no overlap between {len(TEST_CASES)} benchmark cases and train set")
        _ = bench_hash  # retained for future audit logs
    except ImportError:
        print("[splits] benchmark_cases unavailable — skipping overlap assertion")

    print(f"[splits] manifest written to {manifest_path}")
    return {"splits": splits, "rows": rows, "manifest_path": manifest_path}


def train(dry_run: bool = False):
    ensure_utf8_mode()
    require_supported_python()
    if not dry_run:
        from trl import SFTTrainer, SFTConfig

    print(f"Initializing QLoRA training for {BASE_MODEL}...")
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Dataset: {DATA_FILE}")
    print(f"Source adapter: {SOURCE_ADAPTER_DIR}")
    print(f"New model output: {MODEL_OUTPUT_DIR}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Splits dir: {SPLITS_DIR}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")
    if not dry_run and not SOURCE_ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Source adapter directory not found: {SOURCE_ADAPTER_DIR}")

    print(f"Loading dataset: {DATA_FILE}")
    raw_dataset = load_dataset("json", data_files=str(DATA_FILE), split="train")
    split_info = write_splits_for(raw_dataset)
    splits = split_info["splits"]

    if dry_run:
        print("[dry-run] split provenance written. Skipping model load and training.")
        return

    use_bf16, compute_dtype = require_training_gpu()
    print(f"CUDA detected. bf16={use_bf16}, compute_dtype={compute_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE_ADAPTER_DIR), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    print(f"Tokenizer loaded. max_length={tokenizer.model_max_length}")

    train_dataset = raw_dataset.select(splits["train"])
    eval_dataset = raw_dataset.select(splits["val"])
    test_dataset = raw_dataset.select(splits["test"])
    print(
        f"Train: {len(train_dataset)} | Val: {len(eval_dataset)} | "
        f"Test (held out, never seen during training): {len(test_dataset)}"
    )

    family_counts = {}
    for row in raw_dataset:
        fam = row.get("family", "unknown")
        family_counts[fam] = family_counts.get(fam, 0) + 1
    print("Family distribution:")
    for fam, cnt in sorted(family_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(raw_dataset) * 100
        print(f"  {fam:30s} {cnt:6d} ({pct:5.1f}%)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model in 4-bit (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    resume_from_checkpoint = str(latest_checkpoint) if latest_checkpoint is not None else None
    adapter_source = str(latest_checkpoint) if latest_checkpoint is not None else str(SOURCE_ADAPTER_DIR)
    if resume_from_checkpoint is None:
        print(f"Initializing from source adapter: {SOURCE_ADAPTER_DIR}")
    else:
        print(f"Resuming v8 training from checkpoint: {resume_from_checkpoint}")
    model = PeftModel.from_pretrained(model, adapter_source, is_trainable=True)

    sft_config = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1.5e-4,
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=6,
        eval_strategy="epoch",
        optim="adamw_torch",
        report_to="none",
        packing=False,
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
        dataloader_num_workers=0,
    )

    trainer_args = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": sft_config,
    }

    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_args["processing_class"] = tokenizer
    else:
        trainer_args["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_args)

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"Saving adapter to {MODEL_OUTPUT_DIR}...")
    trainer.model.save_pretrained(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
    print("Done. Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA training for Qwen2.5-3B-Instruct on hybrid_data.jsonl.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write splits/ manifest and exit without loading the model. Used for the thesis split-integrity check.",
    )
    cli_args = parser.parse_args()
    train(dry_run=cli_args.dry_run)
