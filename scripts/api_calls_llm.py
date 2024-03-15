from datetime import datetime
# trace_name = f'api_{str(datetime.now())}'

from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_community.llms import LlamaCpp
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage

from operator import itemgetter
import os
# os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
# os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-a45f2a3d-4085-4170-b337-8cc2f1921aef"
# os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-26acbd19-74af-4cb4-93e1-919526c13921"

# import gc
# import torch

# from langfuse import Langfuse
# from langfuse.callback import CallbackHandler
 
# Initialize Langfuse client (prompt management)
# langfuse = Langfuse()
# langfuse_callback_handler = CallbackHandler(trace_name=trace_name)    
# Initialize Langfuse CallbackHandler for Langchain (tracing)
# langfuse_callback_handler = CallbackHandler(
#     secret_key="sk-lf-26acbd19-74af-4cb4-93e1-919526c13921",
#     public_key="pk-lf-a45f2a3d-4085-4170-b337-8cc2f1921aef",
#     host="http://localhost:3000",
# )

 
# Optional, verify that Langfuse is configured correctly
# assert langfuse.auth_check()
# assert langfuse_callback_handler.auth_check()

llm = LlamaCpp(
    # llama 13b saiga: -- '../models/model-q4_K(2).gguf'
    # roleplay - mixtral moe 8x7b: -- mixtral-8x7b-moe-rp-story.Q4_K_M
    # mixtral-8x7b-v0.1.Q4_K_M
    # model_path='../models/mixtral-8x7b-moe-rp-story.Q4_K_M.gguf',
    # n_gpu_layers=8,  # 28 for llama2 13b, 10 for mixtral

    model_path='../models/model-q4_K(2).gguf',
    n_gpu_layers=28,  # 28 for llama2 13b, 10 for mixtral
    max_tokens=500,
    n_batch=128,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    temperature=0.0,
)

# prompt = '''<s> [INST] Сформируй json с ключом name и значением {value} [/INST] </s>'''

# chain = ChatPromptTemplate.from_template(prompt) | chat_model

# print(chain.invoke(input={'value': 'you_see_call'},))

from langchain.chains import APIChain
from langchain.chains.api import open_meteo_docs
from api import payment_docs

# see
# api_request_chain
# api_answer_chain

API_URL_PROMPT_TEMPLATE = """You are given the below API Documentation:
{api_docs}

Using this documentation, generate the full API url to call for using the user's input. Make sure the user's input stays in the original language.
Encode the user's query to be passed over http.
You should build the API url in order to get a response that is as short as possible, while still getting the necessary information to answer the question. Pay attention to deliberately exclude any unnecessary pieces of data in the API call.

User's input:{question}
API url:"""

API_URL_PROMPT = PromptTemplate(
    input_variables=[
        "api_docs",
        "question",
    ],
    template=API_URL_PROMPT_TEMPLATE,
)

API_RESPONSE_PROMPT_TEMPLATE = (
    API_URL_PROMPT_TEMPLATE
    + """ {api_url}

Here is the response from the API:

{api_response}

Summarize this response to answer the original question.

Summary:"""
)

API_RESPONSE_PROMPT = PromptTemplate(
    input_variables=["api_docs", "question", "api_url", "api_response"],
    template=API_RESPONSE_PROMPT_TEMPLATE,
)

api_call = APIChain.from_llm_and_api_docs(
    llm,
    # open_meteo_docs.OPEN_METEO_DOCS,
    payment_docs.PAYMENT_DOCS,
    verbose=True,
    limit_to_domains=["http://localhost:8000/"],
    api_url_prompt=API_URL_PROMPT,
    api_response_prompt=API_RESPONSE_PROMPT,
)

import urllib
def quote(service: str):
    logger.debug(f'before -- {service}')
    logger.debug(f'after -- {urllib.parse.quote(service)}')
    return urllib.parse.quote(service)

chain = RunnableLambda(quote) | api_call

chain.invoke(
    "Пополнение баланса"
    # 'Коммунальные платежи'
)
