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
sys.path.append('/')
from scenarios.base import create_chatml_statement, create_llama3_statement, get_chatml_template, get_llama3_template, parse_markup_chat_history
from llama_cpp import Llama, LlamaGrammar
from typing import Union, Optional
import requests
API_ENDPOINT = "http://localhost:8000/process_text"


# llm = Ollama(
#         model="llama3:instruct",
#         stop=["<|eot_id|>",],
#         num_gpu=33,
#         temperature=0,
#         mirostat=0
#     )


llm = Llama(
        model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q6_K.gguf',
        n_gpu_layers=33,
        max_tokens=1024,
        n_batch=1024,
        n_ctx=2048,
        f16_kv=True,
        verbose=True,
        temperature=0.0,
    )


# template = PromptTemplate.from_template("""<|user|>\n{question} <|end|>\n<|assistant|>""")
app = FastAPI(redirection_slashes=False)

class Input(BaseModel):
    question: str
    chat_history: List[Dict[str, str]]

@app.get("/")
async def hello() -> dict[str, str]:
    return {"hello":"world"}

@app.post("/process_text")
async def process_text(input_data: Input, grammar_path: str = None, stop=['<eot_id>']) -> dict[str, Any]:
    question = input_data.question
    chat_history = input_data.chat_history
    grammar = None
    if grammar_path:
        grammar = LlamaGrammar.from_file(file=grammar_path)


    # phi 3
    # template_str = get_chatml_template(chat_history)

    # llama 3
    template_str = get_llama3_template(SYSTEM_PROMPT_LLAMA3, chat_history)
    # template = PromptTemplate.from_template(template=template_str)
    # chain = template | llm | StrOutputParser()
    logger.debug(template_str)
    # logger.warning(len(template_str))
    result = llm(promt=template_str, grammar=grammar, stop=stop)
    # result = chain.invoke({'question': input_data.question})
    return {"answer": result}


def call_api(input_text: str, chat_history: List[Dict[str, str]]=[]):
    """
    input_text: str
    chat_history: List[Dict[str, str]]
    """
    response = requests.post(
        API_ENDPOINT,
        json={"question": input_text, "chat_history": chat_history}
    )
    return response.json()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)