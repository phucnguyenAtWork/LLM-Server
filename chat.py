"""
FINA Chat — standalone CLI for testing the model locally (no FastAPI).

Loads the base model and all available LoRA adapters at startup, then
lets the operator hot-swap between baseline / v6 / v7 / v8 and toggle
RAG-style "extra context" (a single appended evidence block) without
restarting. Greedy decoding so the same prompt + same config yields a
byte-identical reply, matching the reproducibility claim in Chapter 5.4.

Slash commands:
  /adapter baseline|v6|v7|v8
  /show
  /replay
  /help
  /quit | /exit
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from fina_schema import SYSTEM_PROMPT, parse_model_output, fallback_output

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# v6 trained on Qwen2.5-1.5B; cannot attach to 3B base. See Section 5.3.
ADAPTER_REGISTRY: dict[str, str] = {
    "v7": "financial_qwen_native_v7",
    "v8": "financial_qwen_native_v8",
}
DEFAULT_ADAPTER = "v8"
VALID_ADAPTERS = {"baseline", *ADAPTER_REGISTRY.keys()}


def _print_help():
    print(
        "\nSlash commands:\n"
        "  /adapter baseline|v6|v7|v8\n"
        "  /show       print current config\n"
        "  /replay     re-send the previous prompt\n"
        "  /help       show this list\n"
        "  /quit|/exit leave\n"
    )


def _load():
    print(f"Loading {BASE_MODEL} (8-bit) + adapters {sorted(ADAPTER_REGISTRY)}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )

    first_label = DEFAULT_ADAPTER
    first_path = ADAPTER_REGISTRY[first_label]
    model = PeftModel.from_pretrained(base_model, first_path, adapter_name=first_label)
    for label, path in ADAPTER_REGISTRY.items():
        if label == first_label:
            continue
        if not os.path.isdir(path):
            print(f"[ADAPTER] skip {label}: directory {path!r} not found")
            continue
        try:
            model.load_adapter(path, adapter_name=label)
            print(f"[ADAPTER] loaded {label}")
        except Exception as exc:
            print(f"[ADAPTER] failed to load {label}: {exc}")
    model.set_adapter(DEFAULT_ADAPTER)
    model.eval()
    return tokenizer, model


def _generate(tokenizer, model, messages: list[dict], adapter_label: str) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[-1]

    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        if adapter_label == "baseline":
            with model.disable_adapter():
                outputs = model.generate(**inputs, **gen_kwargs)
        else:
            model.set_adapter(adapter_label)
            outputs = model.generate(**inputs, **gen_kwargs)

    new_ids = outputs[0][input_length:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _print_response(raw: str, cfg: dict):
    parsed = parse_model_output(raw)
    if parsed is None:
        parsed = fallback_output(raw)
        print(f"\n[adapter={cfg['adapter']}] (raw output, parse failed)")
        print(f"FINA (raw): {raw[:500]}")
        return
    print(f"\n[adapter={cfg['adapter']}]")
    print(f"FINA: {parsed.message}")
    if parsed.action:
        action_dict = {
            "type": parsed.action.type if hasattr(parsed.action, "type") else None,
            "arguments": parsed.action.arguments.model_dump() if parsed.action.arguments else None,
        }
        print(f"  Action: {json.dumps(action_dict, ensure_ascii=False, default=str)}")
    if parsed.signals:
        print(f"  Signals: {parsed.signals}")


def chat():
    tokenizer, model = _load()

    print("\nFINA ONLINE (JSON mode, greedy). Type /help for commands.")
    print("-" * 50)

    history: list[dict] = []
    cfg = {"adapter": DEFAULT_ADAPTER}
    last_prompt: str | None = None

    while True:
        try:
            user_input = input(f"\n[adapter={cfg['adapter']}] User: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        line = user_input.strip()
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("/quit", "/exit"):
                break
            if cmd == "/help":
                _print_help()
                continue
            if cmd == "/show":
                print(f"  adapter={cfg['adapter']}")
                continue
            if cmd == "/adapter":
                if arg.lower() in VALID_ADAPTERS:
                    cfg["adapter"] = arg.lower()
                    print(f"  adapter -> {cfg['adapter']}")
                else:
                    print(f"  invalid adapter. choose from: {sorted(VALID_ADAPTERS)}")
                continue
            if cmd == "/replay":
                if last_prompt is None:
                    print("  no previous prompt to replay")
                    continue
                line = last_prompt
                print(f"  replaying: {last_prompt!r}")
            else:
                print(f"  unknown command: {cmd}. type /help.")
                continue

        last_prompt = line
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                    {"role": "user", "content": line}]
        raw = _generate(tokenizer, model, messages, cfg["adapter"])
        _print_response(raw, cfg)
        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": raw})


if __name__ == "__main__":
    chat()
