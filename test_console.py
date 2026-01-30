import requests
import json

API_URL = "http://localhost:8105/chat"

def start_console_chat():
    print("\n" + "="*60)
    print(" FINA TERMINAL CLIENT (Testing Mode)")
    print("    Connected to: " + API_URL)
    print("    Type 'quit' to exit.")
    print("="*60 + "\n")
    current_user_id=1
    role = "Student"
    mode = "Survival" 

    while True:
        # 1. Get User Input
        user_msg = input(f"\n[User {current_user_id} | {mode}] You: ")
        
        if user_msg.lower() in ["quit", "exit"]:
            print("Exiting...")
            break
        
        payload = {
            "user_id": current_user_id,
            "role": role,
            "mode": mode,
            "message": user_msg
        }

        try:
            # 3. Send to api.py
            print(" ... Thinking ...")
            response = requests.post(API_URL, json=payload)
            
            # 4. Print Response
            if response.status_code == 200:
                data = response.json()
                print(f"\nFINA: {data['response']}")
            else:
                print(f" Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            print(" Error: Could not connect to api.py. Is it running?")

if __name__ == "__main__":
    start_console_chat()