import requests
import sys
sys.path.append('/home/amstel/llm/src')
from scenarios.base import get_llama3_template

prompt = get_llama3_template(system_prompt_clean="You are a great AI assistant.", chat_history=[])
print(prompt)

resp = requests.post(url="127.0.0.1:8000/process_text", json={"question": prompt, "chat_history": []})
print("response: ")
print(resp)
print(resp.json())