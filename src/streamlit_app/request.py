import requests
import sys
sys.path.append('/home/amstel/llm/src')
# from scenarios.base import get_llama3_template

# prompt = get_llama3_template(system_prompt_clean="You are a great AI assistant.", chat_history=[{'user':'Hey whats going on man?'}])
# print(prompt)
if __name__ == '__main__':

    resp = requests.post(url="http://127.0.0.1:8000/process_text", json={
        "question": '',
        "chat_history": [{'role':'user', 'content':'Hello!'}],
        "grammar_path": '',
        "stop": ["<eot_id>"],
    })
    print("response: ")
    print(resp)
    print(resp.json())