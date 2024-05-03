from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Dict
import uvicorn
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
SYSTEM_PROMPT_LLAMA3 = "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability."

# instead of the functios below we can extend:
# https://api.python.langchain.com/en/latest/_modules/langchain_experimental/chat_models/llm_wrapper.html#Llama2Chat
def create_chatml_statement(role: str, content: str):
    assert role in ('user', 'assistant')
    assert content
    return f"<|{role}|>\n{content}<|end|>\n"

def create_llama3_statement(role: str, content: str):
    assert role in ('user', 'assistant')
    assert content
    return f"<|start_header_id|>{ role }<|end_header_id|>\n{ content }<|eot_id|>"

def get_chatml_template(chat_history: List[Dict[str, str]]):
    '''chat history ~ few shot'''
    template = ""
    final_assistant = "<|assistant|>\n"
    if chat_history:
        logger.warning(chat_history)
        for message in chat_history:
            role = message.get('role')
            content = message.get('content')
            current_template_part = create_chatml_statement(role, content)
            template += current_template_part
        template += final_assistant
        logger.warning(template)
    else:
        raise AttributeError  # should never be executed
        # final_user = create_statement('user', question)
        # template = final_user + final_assistant
        # logger.warning(template)
    return template

def get_llama3_template(system_prompt_clean:str, chat_history: List[Dict[str, str]]):
    """"""
    assert chat_history
    final_assistant = "<|start_header_id|>assistant<|end_header_id|>"
    template = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{ system_prompt_clean }<|eot_id|>'
    for message in chat_history:
        role = message.get('role')
        content = message.get('content')
        current_template_part = create_llama3_statement(role, content)
        template += current_template_part
    template += final_assistant
    logger.warning(template)
    return template


llm = LlamaCpp(
    # model_path='/home/amstel/llm/models/publisher/repository/Phi-3-mini-4k-instruct-q4.gguf',
    # stop=['<|end|>'], # phi

    model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf',  # good
    # model_path='/home/amstel/llm/models/publisher/repository/saiga_llama3_8b_q4_k.gguf',
    stop=["<|eot_id|>", "<|start_header_id|>", '```', '```\n', ],

    n_gpu_layers=33,
    temperature=0.0,
    max_tokens=1024,
    n_batch=256,
    n_ctx=2048,
    f16_kv=True,
    verbose=True,
)
# template = PromptTemplate.from_template("""<|user|>\n{question} <|end|>\n<|assistant|>""")
app = FastAPI(redirection_slashes=False)

# def log_data(x):
#     logger.debug(x)
#     return x
#
# logger_runnable = RunnablePassthrough(log_data)

class Input(BaseModel):
    question: str
    chat_history: List[Dict[str, str]]

@app.get("/")
async def hello() ->dict[str, str]:
    return {"hello":"world"}

@app.post("/process_text")
async def process_text(input_data: Input) -> dict[str, Any]:
    question = input_data.question
    chat_history = input_data.chat_history

    # phi 3
    # template_str = get_chatml_template(chat_history)

    # llama 3
    template_str = get_llama3_template(SYSTEM_PROMPT_LLAMA3, chat_history)

    template = PromptTemplate.from_template(template=template_str)
    chain = template | llm | StrOutputParser()
    logger.debug(template)
    logger.warning(len(template_str))

    result = chain.invoke({'question': input_data.question})
    return {"answer": result}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)