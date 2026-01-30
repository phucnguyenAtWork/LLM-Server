import torch
import os
import pymysql.cursors 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn

# ==========================================
# CONFIGURATION
# ==========================================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_NAME = "financial_qwen_native_v1"

MAC_IP = "100.109.225.15"

# Connection Config
DB_CONFIG = {
    "host": MAC_IP,
    "port": 3308,     
    "user": "root",
    "password": "rootpass",
    "database": "financedb",
    "cursorclass": pymysql.cursors.DictCursor
}

SYSTEM_PROMPT = """
You are FINA, an advanced Data-Driven Financial AI, designed to act as a sophisticated personal CFO.
Your goal is to analyze the user's financial data with the depth and clarity of a professional analyst.

### YOUR INSTRUCTIONS:
1. **Analyze, Don't Just Report:** Do not simply repeat the numbers. Interpret them. (e.g., "You have spent $200 on food, which is 40% of your budget. This is high.")
2. **Context-Aware Advice:**
   - If **Mode = Survival**: Be urgent and protective. Focus strictly on cutting costs to $0. Ignore "Savings Targets" and focus on "Cash Flow."
   - If **Mode = Savings**: Be strict about the 50/30/20 rule. Highlight exactly where they are over-spending.
   - If **Mode = Standard**: Be balanced and conversational.
3. **The 50/30/20 Rule:** Use the calculated "Budget Guidelines" in the context to grade the user's performance.

### RESPONSE STRUCTURE:
- **The Snapshot:** A 1-sentence summary of their current financial health.
- **The Analysis:** Compare their Actual Spending vs. the Target. Highlight gaps.
- **The Action Plan:** Give 2-3 specific, actionable steps they can take *right now*.

### TONE:
Professional, empathetic, intelligent, and concise. Avoid robotic language.
"""

app = FastAPI(title="FINA Financial AI API")

# Global variables
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("API STARTUP: Loading FINA Brain...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_NAME)
    model.eval()
    print("API READY: FINA is online.")

def get_financial_summary(user_id):
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            
            sql_income = "SELECT SUM(amount) as total FROM transactions WHERE user_id = %s AND type = 'INCOME'"
            cursor.execute(sql_income, (user_id,))
            res_income = cursor.fetchone()
            income = float(res_income['total']) if res_income and res_income['total'] else 0.0

            sql_spending = """
                SELECT c.name as category_name, SUM(t.amount) as spent 
                FROM transactions t
                JOIN categories c ON t.category_id = c.id
                WHERE t.user_id = %s AND t.type = 'EXPENSE'
                GROUP BY c.name
            """
            cursor.execute(sql_spending, (user_id,))
            spending_data = cursor.fetchall() 
            budget_needs = income * 0.50
            budget_wants = income * 0.30
            budget_savings = income * 0.20

        connection.close()
        summary = f"""
        --- FINANCIAL CONTEXT ---
        USER INCOME: ${income:.2f}
        
        CURRENT SPENDING (Actual):
        {spending_data}
        
        TARGET BUDGETS (50/30/20 Rule):
        - Needs Limit (Rent, Food): ${budget_needs:.2f}
        - Wants Limit (Fun, Shopping): ${budget_wants:.2f}
        - Savings Target: ${budget_savings:.2f}
        ---------------------------
        """
        return summary

    except Exception as e:
        print(f"DB Error: {e}")
        return "Error: Could not fetch financial data."

class UserRequest(BaseModel):
    user_id: int = 1
    role: str = "Student"
    mode: str=["Standard","Survival","Savings"]
    message: str           

@app.post("/chat")
async def chat_endpoint(request: UserRequest):
    global model, tokenizer
    
    financial_context = get_financial_summary(user_id=request.user_id)
    print("\n" + "="*50)
    print(f"FINA BRAIN LOG (Incoming Request)")
    print(f"Role: {request.role} | Mode: {request.mode}")
    print(f"Data Fetched: {financial_context}") 
    print(f"Question: {request.message}")
    print("="*50 + "\n")
    
    # 2. CONSTRUCT PROMPT
    full_message = (
        f"User Role: {request.role}\n"
        f"Selected Mode: {request.mode}\n"
        f"Financial Context: {financial_context}\n"
        f"User Question: {request.message}"
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_message}
    ]
    
    # 3. GENERATE
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean_response = response.split("assistant")[-1].strip() if "assistant" in response else response

    return {"response": clean_response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8105)