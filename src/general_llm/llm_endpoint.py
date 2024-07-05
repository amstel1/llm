MODEL_NAME = 'llama'
assert MODEL_NAME in ('llama', 'gemma')
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

import sys
sys.path.append('/home/amstel/llm/src')
from general_llm.prompt_construction import Llama3PromptTemplate, Gemma2PromptTemplate, parse_markup_chat_history
from llama_cpp import Llama, LlamaGrammar
from typing import Union, Optional
import requests
from llama_cpp.llama_types import CreateCompletionResponse, CreateCompletionStreamResponse

class LLMEndpointInput(BaseModel):
    user_prompt: str = None # assert it's not None
    system_prompt: str = None # assert it's not None
    assistant_must_start_with: str = ""
    chat_history: List[Dict[str, str]] = None
    grammar_path: str | None = None
    stop: List[str] = []


# todo: exctract two defs below into RPI layer
# start PRI layer
def call_generate_from_history_api(
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        assistant_must_start_with: str = "",
        grammar_path: str = None,
        stop: List[str] = [],
) -> str:
    """
    input_text: str
    chat_history: List[Dict[str, str]]
    """
    logger.debug({
            "system_prompt": system_prompt,
            "chat_history": chat_history,
            "assistant_must_start_with": assistant_must_start_with,
            "grammar_path": grammar_path,
            "stop": stop})
    response = requests.post(
        'http://localhost:8000/generate-from-history',
        json={
            "system_prompt": system_prompt,
            "chat_history": chat_history,
            "assistant_must_start_with": assistant_must_start_with,
            "grammar_path": grammar_path,
            "stop": stop}
    )

    r = response.json().get('choices')[0].get('text')
    return r


def call_generate_from_query_api(
        user_prompt: str,
        system_prompt: str,
        assistant_must_start_with: str = "",
        grammar_path: str = None,
        stop: List[str] = [],) -> str:
    """
    input_text: str
    """
    logger.info(f'reformulate :: call_generate_from_query_api - user propmpt: {user_prompt}')
    logger.info(f'reformulate :: call_generate_from_query_api - system propmpt: {system_prompt}')
    logger.info(f'reformulate :: call_generate_from_query_api - grammar_path: {grammar_path}')

    response = requests.post(
        'http://localhost:8000/generate-from-query',
        json={"user_prompt": user_prompt, "system_prompt": system_prompt, "assistant_must_start_with": assistant_must_start_with, "grammar_path": grammar_path, "stop": stop}
    )
    r = response.json().get('choices')[0].get('text')
    logger.info(f'reformulate :: call_generate_from_query_api - response: {r}')

    return r

def call_generation_api(prompt: str, grammar: str = None, grammar_path: str = None, stop: list = None) -> str:
    '''completion API'''
    logger.critical(f'2606 - endpoint - debug: {prompt}')
    logger.critical(f'{len(prompt)}')
    response = requests.post(
        'http://localhost:8000/generate',
        json={"prompt": prompt, "grammar": grammar, "stop": stop}
    )
    r = response.json().get('choices')[0].get('text')
    return r
# end RPI layer


app = FastAPI(redirection_slashes=False)

if 'llama' in MODEL_NAME.lower():
    @app.on_event("startup")
    async def load_llm():
        global llm
        llm = Llama(
            model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q6_K.gguf',
            n_gpu_layers=33,
            max_tokens=-1,
            n_batch=512,
            n_ctx=8192,
            f16_kv=True,
            verbose=True,
            temperature=0.0,
            flash_attn=True,
        )
        global eot_list
        eot_list = ['<|eot_id|>']

        @app.get("/")
        async def hello() -> dict[str, str]:
            return {"model_name": "llama3"}
elif 'gemma' in MODEL_NAME.lower():
    @app.on_event("startup")
    async def load_llm():
        global llm
        llm = Llama(
            model_path='/home/amstel/llm/models/bartowski/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q5_K_S.gguf',
            n_gpu_layers=43,
            max_tokens=-1,
            n_batch=128,
            n_ctx=4096,
            f16_kv=True,
            verbose=True,
            temperature=0.0,
            flash_attn=True,
        )
        global eot_list
        eot_list = ['<end_of_turn>']

        @app.get("/")
        async def hello() -> dict[str, str]:
            return {"model_name": "gemma2"}

@app.post("/generate-from-history")
async def generate_from_history(input_data: LLMEndpointInput) -> Dict[str, Any]:

    system_prompt = input_data.system_prompt
    chat_history = input_data.chat_history
    assistant_must_start_with = input_data.assistant_must_start_with
    grammar_path = input_data.grammar_path
    stop = input_data.stop
    if not stop:
        stop = eot_list # todo: это кусок дерьма, а что если модель не ллама
    else:
        assert isinstance(stop, list)
        stop.extend(eot_list)
    grammar = None
    if grammar_path and grammar_path.endswith('.gbnf'):
        grammar = LlamaGrammar.from_file(file=grammar_path)

    if 'llama' in MODEL_NAME: prompt_template = Llama3PromptTemplate
    if 'gemma' in MODEL_NAME: prompt_template = Gemma2PromptTemplate
    prompt_str = prompt_template().create_prompt_from_history(
        system_prompt_clean=system_prompt,
        chat_history=chat_history,
        assistant_must_start_with=assistant_must_start_with
    )
    logger.debug(f"1106 debug chatting: {prompt_str.replace('<|begin_of_text|>', '')}")
    result = llm(
        prompt=prompt_str.replace('<|begin_of_text|>', ''),
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
    user_prompt = input_data.user_prompt.replace('<|begin_of_text|>', '')
    system_prompt = input_data.system_prompt.replace('<|begin_of_text|>', '')
    assistant_must_start_with = input_data.assistant_must_start_with
    grammar_path = input_data.grammar_path
    stop = input_data.stop
    if not stop:
        stop = eot_list # todo: это кусок дерьма, а что если модель не ллама
    else:
        assert isinstance(stop, list)
        stop.extend(eot_list)
    grammar = None
    if grammar_path:
        grammar = LlamaGrammar.from_file(file=grammar_path)

    # todo: llama 3
    if 'llama' in MODEL_NAME: prompt_template = Llama3PromptTemplate
    if 'gemma' in MODEL_NAME: prompt_template = Gemma2PromptTemplate
    prompt_str = prompt_template().create_prompt_from_user_query(
        system_prompt_clean=system_prompt,
        user_query=user_prompt,
        assistant_must_start_with=assistant_must_start_with
    )
    logger.warning(f'stop: {type(stop)}, {stop}')
    logger.debug(f'{type(prompt_str)} -- {prompt_str}')
    result = llm(
        prompt=prompt_str,
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
    grammar_path = input.get('grammar_path')
    stop = input.get('stop')
    if not stop:
        stop = eot_list
    else:
        assert isinstance(stop, list)
        stop.extend(eot_list)
    if grammar and not grammar_path:
        grammar = LlamaGrammar.from_string(grammar=grammar)
    if not grammar and grammar_path and grammar_path.endswith('.gbnf'):
        grammar = LlamaGrammar.from_file(file=grammar_path)
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