"""
FINA Chat — standalone CLI for testing the model locally.
Uses the same SYSTEM_PROMPT and JSON output format as api.py.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from fina_schema import SYSTEM_PROMPT, parse_model_output, fallback_output

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_NAME = "financial_qwen_native_v7"


def chat():
    print(f"Loading {BASE_MODEL} + {ADAPTER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_NAME)
    model.eval()

    print("\nFINA ONLINE (JSON mode). Type 'quit' to exit.")
    print("-" * 50)

    history = []

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_length = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_ids = outputs[0][input_length:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        parsed = parse_model_output(raw)
        if parsed:
            print(f"\nFINA: {parsed.message}")
            if parsed.action:
                print(f"  Action: {parsed.action.type} {parsed.action.arguments.model_dump()}")
            if parsed.signals:
                print(f"  Signals: {parsed.signals}")
        else:
            print(f"\nFINA (raw): {raw}")

        # Track history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": raw})


if __name__ == "__main__":
    chat()
