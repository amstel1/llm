from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Dict
import uvicorn
from langchain_community.llms import Ollama, LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
SYSTEM_PROMPT_LLAMA3 = "Ты - общительный, вежливый, внимательный, интересный собеседник. Ты выполняешь просьбы клиента наилучшим образом. Ты знаешь только русский язык."
import sys
sys.path.append('/home/amstel/llm/src')
from scenarios.base import create_chatml_statement, create_llama3_statement, get_chatml_template, get_llama3_template, parse_markup_chat_history
from llama_cpp import Llama, LlamaGrammar
from typing import Union, Optional
import requests


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
    grammar_path: str = None
    stop: List[str] = ['<eot_id>']

@app.on_event("startup")
async def load_llm():
    global llm
    llm = Llama(
        model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B.Q2_K.gguf',
        n_gpu_layers=20,
        max_tokens=1024,
        n_batch=1024,
        n_ctx=2048,
        f16_kv=True,
        verbose=True,
        temperature=0.0,
    )
    logger.info("LLM Loaded")

@app.get("/")
async def hello() -> dict[str, str]:
    return {"hello": "world"}

@app.post("/process_text")
async def process_text(input_data: Input) -> dict[str, Any]:
    logger.debug(input_data)
    question = input_data.question
    chat_history = input_data.chat_history
    grammar_path = input_data.grammar_path
    stop = input_data.stop
    grammar = None
    if grammar_path:
        grammar = LlamaGrammar.from_file(file=grammar_path)

    # phi 3
    # template_str = get_chatml_template(chat_history)

    # llama 3
    template_str = get_llama3_template(SYSTEM_PROMPT_LLAMA3, chat_history)
    logger.debug(template_str)
    result = llm(prompt=template_str, grammar=grammar, stop=stop, temperature=0, echo=True)
    return result




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)