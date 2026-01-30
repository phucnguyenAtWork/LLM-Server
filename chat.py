# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# # ==========================================
# # CONFIGURATION
# # ==========================================
# BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# ADAPTER_NAME = "financial_qwen_native_v1"

# SYSTEM_PROMPT = """
# You are FINA, an advanced Financial AI Assistant. Your goal is to provide accurate, role-specific financial guidance.

# WHO YOU ARE:
# - You act as both a strict "Secretary" (auditing budgets) and a strategic "Advisor" (giving life advice).
# - You are professional, concise, and mathematically precise.

# YOUR CAPABILITIES:
# 1. Budget Auditing: Analyze 'Target' vs 'Actual' spending. Flag overspending immediately.
# 2. Role-Based Advice: adapt your strategy based on the user's role:
#    - STUDENTS: Focus on cutting costs, avoiding debt, and student discounts.
#    - FREELANCERS: Focus on tax savings (30% rule), income stability, and business deductions.
#    - WORKERS: Focus on 401k matching, emergency funds, and long-term investing.

# CONSTRAINTS:
# - Do NOT hallucinate. If you do not know a financial term, ask for clarification.
# - Keep answers under 4 sentences unless asked for a detailed report.
# """

# def chat():
#     print(" Loading FINA (Financial AI)...")
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
#     base_model = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL,
#         dtype=torch.float16,
#         device_map="auto",
#         trust_remote_code=True
#     )

#     print(f"Attaching Brain: {ADAPTER_NAME}...")
#     model = PeftModel.from_pretrained(base_model, ADAPTER_NAME)
#     model.eval()

#     print("\nFINA ONLINE. (Type 'quit' to exit)")
#     print("-" * 50)

#     while True:
#         user_input = input("\nUser: ")
#         if user_input.lower() in ["quit", "exit"]:
#             break

#         # Combine System Prompt + User Question
#         messages = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": user_input}
#         ]
        
#         # Format for Qwen
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         inputs = tokenizer(text, return_tensors="pt").to("cuda")

#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=256,
#                 temperature=0.3,
#                 top_p=0.9,
#                 do_sample=True,
#                 pad_token_id=tokenizer.eos_token_id
#             )
        
#         # Clean Output
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         if "assistant" in response:
#             clean_response = response.split("assistant")[-1].strip()
#         else:
#             clean_response = response
            
#         print(f"FINA: {clean_response}")

# if __name__ == "__main__":
#     chat()