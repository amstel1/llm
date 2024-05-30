import json

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Dict, Iterator
import uvicorn
from langchain_community.llms import Ollama, LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
SYSTEM_PROMPT_LLAMA3 = "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability."
import sys
sys.path.append('/home/amstel/llm/src')
from scenarios.base import create_chatml_statement, create_llama3_statement, get_chatml_template, get_llama3_template_from_history, get_llama3_template_from_user_query, parse_markup_chat_history
from llama_cpp import Llama, LlamaGrammar
from typing import Union, Optional
import requests
from llama_cpp.llama_types import CreateCompletionResponse, CreateCompletionStreamResponse

class LLMEndpointInput(BaseModel):
    user_prompt: str = None # assert it's not None
    system_prompt: str = None # assert it's not None
    chat_history: List[Dict[str, str]] = None
    grammar_path: str | None = None
    stop: List[str] = ['<|eot_id|>']


# todo: exctract two defs below into RPI layer
# start PRI layer
def call_generate_from_history_api(
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        grammar_path: str = None,
        stop: List[str] = ['<|eot_id|>'],
) -> str:
    """
    input_text: str
    chat_history: List[Dict[str, str]]
    """
    response = requests.post(
        'http://localhost:8000/generate-from-history',
        json={"system_prompt": system_prompt, "chat_history": chat_history, "grammar_path": grammar_path, "stop": stop}
    )
    r = response.json().get('choices')[0].get('text')
    return r


def call_generate_from_query_api(
        user_prompt: str,
        system_prompt: str,
        grammar_path: str = None,
        stop: List[str] = ['<|eot_id|>'],) -> str:
    """
    input_text: str
    """
    response = requests.post(
        'http://localhost:8000/generate-from-query',
        json={"user_prompt": user_prompt, "system_prompt": system_prompt, "grammar_path": grammar_path, "stop": stop}
    )
    r = response.json().get('choices')[0].get('text')
    return r

def call_generation_api(prompt: str, grammar: str = None, stop: list = None) -> str:
    response = requests.post(
        'http://localhost:8000/generate',
        json={"prompt": prompt, "grammar": grammar, "stop": stop}
    )
    r = response.json().get('choices')[0].get('text')
    return r
# end RPI layer


app = FastAPI(redirection_slashes=False)


@app.on_event("startup")
async def load_llm():
    global llm
    llm = Llama(
        # alt: Meta-Llama-3-8B-Instruct-Q6_K.gguf, Llama-3-8B-Instruct-32k-v0.1.Q6_K.gguf, LLaMA3-iterative-DPO-final-Q4_K_M.gguf
        model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q6_K.gguf',
        n_gpu_layers=33,
        max_tokens=-1,
        n_batch=512,
        n_ctx=2048,
        f16_kv=False,
        verbose=True,
        temperature=0.0,
    )


@app.get("/")
async def hello() -> dict[str, str]:
    return {"hello": "world"}


@app.post("/generate-from-history")
async def generate_from_listory(input_data: LLMEndpointInput) -> Dict[str, Any]:
    system_prompt = input_data.system_prompt
    chat_history = input_data.chat_history
    grammar_path = input_data.grammar_path
    stop = input_data.stop
    grammar = None
    if grammar_path and grammar_path.endswith('.gbnf'):
        grammar = LlamaGrammar.from_file(file=grammar_path)

    # phi 3
    # template_str = get_chatml_template(chat_history)

    # llama 3
    template_str = get_llama3_template_from_history(system_prompt_clean=system_prompt, chat_history=chat_history)
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


@app.post("/generate-from-query")
async def generate_from_query(input_data: LLMEndpointInput) -> Dict[str, Any]:
    user_prompt = input_data.user_prompt
    system_prompt = input_data.system_prompt
    grammar_path = input_data.grammar_path
    stop = input_data.stop
    grammar = None
    if grammar_path:
        grammar = LlamaGrammar.from_file(file=grammar_path)

    # llama 3
    template_str = get_llama3_template_from_user_query(system_prompt_clean=system_prompt, user_query=user_prompt)
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
async def generate(input: dict) -> Dict[str, Any]:
    # output str?
    """input keys: prompt (required), grammar (optional), stop (optional) """
    prompt = input.get('prompt')
    grammar = input.get('grammar')
    stop = input.get('stop')
    if not stop:
        stop = ['<|eot_id|>']
    if grammar:
        grammar = LlamaGrammar.from_string(grammar=grammar)
    result = llm(
        prompt=prompt,
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)