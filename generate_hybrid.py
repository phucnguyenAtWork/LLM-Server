import json
import random
from datasets import load_dataset

OUTPUT_FILE = "hybrid_data.jsonl"
NUM_SYNTHETIC = 1000
NUM_ALPACA = 2000      

def format_qwen(prompt, response):
    # This is the standard format for Qwen/DeepSeek/OpenAI-compatible models
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

def fetch_finance_alpaca(n=2000):
    print(f"⬇️ Downloading 'gbharti/finance-alpaca' dataset...")
    # Load dataset from Hugging Face
    ds = load_dataset("gbharti/finance-alpaca", split="train")
    
    # Shuffle and pick N examples
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    
    processed_data = []
    for row in ds:
        user_text = row['instruction']
        if row.get('input'):
            user_text += f"\nContext: {row['input']}"
            
        assistant_text = row['output']
        
        formatted = format_qwen(user_text, assistant_text)
        processed_data.append({"text": formatted})
        
    print(f"Loaded {len(processed_data)} examples from Finance-Alpaca.")
    return processed_data


FORMULAS = {
    "Student": {"Needs": 50, "Wants": 30, "Savings": 20},
    "Freelancer": {"Needs": 40, "Tax": 30, "Business": 20, "Savings": 10},
    "Worker": {"Needs": 50, "Wants": 20, "Savings": 30}
}

ADVICE_KNOWLEDGE = {
    "Student": "Focus on cutting 'Wants'. Use student discounts. Do not take credit card debt.",
    "Freelancer": "Your priority is TAXES. Save 30% of every check immediately. Create an income buffer.",
    "Worker": "Maximize employer 401k matching. Build a 6-month emergency fund."
}

def generate_secretary_task():
    role = random.choice(list(FORMULAS.keys()))
    targets = FORMULAS[role]
    
    # 50/50 Chance of Pass/Fail
    if random.choice([True, False]):
        current = {k: v + random.randint(-2, 2) for k, v in targets.items()}
        response = "Status: PASS. You are strictly adhering to the formula."
    else:
        current = targets.copy()
        current["Needs"] += 15
        current["Savings"] -= 15
        response = f"ALERT: Deviation detected. You are overspending on Needs ({current['Needs']}% vs Target {targets['Needs']}%)."

    prompt = f"Task: Budget Audit\nProfile: {role}\nTarget: {json.dumps(targets)}\nActual: {json.dumps(current)}\nAnalyze compliance."
    return format_qwen(prompt, response)

def generate_advisor_task():
    role = random.choice(list(ADVICE_KNOWLEDGE.keys()))
    problems = ["I want to buy a car.", "Should I invest?", "I'm broke."]
    problem = random.choice(problems)
    
    prompt = f"Task: Strategic Advice\nProfile: {role}\nProblem: {problem}\nGive me guidance."
    response = f"As a {role}, remember: {ADVICE_KNOWLEDGE[role]} Regarding '{problem}': prioritize safety net first."
    return format_qwen(prompt, response)

# ==========================================
if __name__ == "__main__":
    final_dataset = []
    # 1. Fetch dataset
    final_dataset.extend(fetch_finance_alpaca(NUM_ALPACA))

    # 2. Generate Synthetic Data
    print(f" Generating {NUM_SYNTHETIC} synthetic examples...")
    for _ in range(NUM_SYNTHETIC // 2):
        final_dataset.append({"text": generate_secretary_task()})
        final_dataset.append({"text": generate_advisor_task()})

    # 3. Shuffle everything together
    random.shuffle(final_dataset)

    # 4. Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in final_dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"Success! Saved {len(final_dataset)} training examples to '{OUTPUT_FILE}'.")