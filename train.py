"""
FINA Training Script v3
========================
QLoRA fine-tuning for Qwen2.5-3B-Instruct using structured prompt/completion data.
Supervises only assistant completions, not the full prompt.

Dataset format (JSONL):
  {"prompt": [...messages...], "completion": [...messages...], "family": "..."}

Usage: python train.py
"""

import inspect
import os
import sys
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
MAX_SEQ_LENGTH = 2048
EVAL_SPLIT = 0.05  # 5% held out for validation


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


def train():
    ensure_utf8_mode()
    require_supported_python()
    from trl import SFTTrainer, SFTConfig

    print(f"Initializing QLoRA training for {BASE_MODEL}...")
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Dataset: {DATA_FILE}")
    print(f"Source adapter: {SOURCE_ADAPTER_DIR}")
    print(f"New model output: {MODEL_OUTPUT_DIR}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")
    if not SOURCE_ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Source adapter directory not found: {SOURCE_ADAPTER_DIR}")

    use_bf16, compute_dtype = require_training_gpu()
    print(f"CUDA detected. bf16={use_bf16}, compute_dtype={compute_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE_ADAPTER_DIR), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    print(f"Tokenizer loaded. max_length={tokenizer.model_max_length}")

    print(f"Loading dataset: {DATA_FILE}")
    raw_dataset = load_dataset("json", data_files=str(DATA_FILE), split="train")

    split = raw_dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

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
    train()
