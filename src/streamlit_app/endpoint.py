from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Dict
import uvicorn
from langchain_community.llms import Ollama, LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
SYSTEM_PROMPT_LLAMA3 = "You are a helpful assistant."
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
        model_path='/home/amstel/llm/models/Publisher/Repository/Llama-3-8B-Instruct-32k-v0.1.Q6_K.gguf',
        n_gpu_layers=33,
        max_tokens=-1,
        n_batch=512,
        n_ctx=2048,
        f16_kv=False,
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
    result = llm(
        prompt=template_str,
        grammar=grammar,
        stop=stop,
        echo=False,
        max_tokens=-1,
        suffix=None,
        temperature=0.0,
        top_p=0.95,
        min_p=0.05,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.1,
        top_k=40,
        mirostat_mode=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
    )
    return result

@app.post("/generate")
async def generate(input: dict) -> dict[str, Any]:
    prompt = input.get('prompt')
    grammar = input.get('grammar')
    if grammar:
        grammar = LlamaGrammar.from_string(grammar=grammar)
    logger.critical(grammar)
    result = llm(
        prompt=prompt,
        grammar=grammar,
        stop=['<|eot_id|>'],
        echo=False,
        max_tokens=-1,
        suffix=None,
        temperature=0.0,
        top_p=0.95,
        min_p=0.05,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.1,
        top_k=40,
        mirostat_mode=0,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
    )
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)