import os
import torch
import inspect
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ==========================================
# CONFIGURATION
# ==========================================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NEW_MODEL_NAME = "financial_qwen_native_v1"
DATA_FILE = "hybrid_data.jsonl"
MAX_SEQ_LENGTH = 512

def train():
    print(f" Initializing Training for {BASE_MODEL}...")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    print(f"Enforced max sequence length: {tokenizer.model_max_length}")

    # 2. Load Dataset
    print(f"Loading dataset: {DATA_FILE}")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # 3. Load Model (Native FP16)
    print("Loading model in Float16...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Enable memory saving
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 4. Configure LoRA (The Adapter)
    # The Trainer will apply them for us.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 5. Initialize SFTConfig
    sft_config = SFTConfig(
        output_dir=f"./{NEW_MODEL_NAME}_checkpoints",
        dataset_text_field="text",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        report_to="none",
        packing=False,
    )

    # 6. Prepare Trainer Arguments
    trainer_args = {
        "model": model,             # Pass the RAW base model
        "train_dataset": dataset,
        "peft_config": peft_config, # Pass config so Trainer handles wrapping
        "args": sft_config,
    }

    # 7. Auto-Detect Correct Tokenizer Argument
    # This prevents the "unexpected keyword argument" error
    trainer_signature = inspect.signature(SFTTrainer.__init__)
    if 'processing_class' in trainer_signature.parameters:
        print("Detected: Library wants 'processing_class'")
        trainer_args['processing_class'] = tokenizer
    else:
        print("Detected: Library wants 'tokenizer'")
        trainer_args['tokenizer'] = tokenizer

    # 8. Start Trainer
    trainer = SFTTrainer(**trainer_args)

    print("Starting Training...")
    trainer.train()

    # 9. Save
    print(f"Saving model to {NEW_MODEL_NAME}...")
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)
    print("DONE! Model saved.")

if __name__ == "__main__":
    train()