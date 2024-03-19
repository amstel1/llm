from datetime import datetime
trace_name = f'deposit_{str(datetime.now())}'

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
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-a45f2a3d-4085-4170-b337-8cc2f1921aef"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-26acbd19-74af-4cb4-93e1-919526c13921"

# import gc
# import torch

from langfuse import Langfuse
from langfuse.callback import CallbackHandler
 
# Initialize Langfuse client (prompt management)
langfuse = Langfuse()
langfuse_callback_handler = CallbackHandler(trace_name=trace_name)    

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_callback_handler = CallbackHandler(
    secret_key="sk-lf-26acbd19-74af-4cb4-93e1-919526c13921",
    public_key="pk-lf-a45f2a3d-4085-4170-b337-8cc2f1921aef",
    host="http://localhost:3000",
)

 
# Optional, verify that Langfuse is configured correctly
# assert langfuse.auth_check()
assert langfuse_callback_handler.auth_check()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_docs_for_query(query: str) -> List[str]:
    ''''''
    assert isinstance(query, str)
    # fake retrieval result
    loader = TextLoader("../documents/vklady--v-belorusskih-rublyah_full.txt")
    docs = loader.load()
    # logger.warning(docs)
    return docs

dummy_retriever = RunnableLambda(retrieve_docs_for_query)


physics_template = """ Ты - профессор физики. Ты специализируешься на квантовой механике и термодинамике. \
При ответе учитывай историю. Кратко отвечай на вопросы. \
Отвечай на вопрос ровно один раз.

История:
{history}.

Вопрос:
{query}


Ответ: """

math_template = """ Ты - математик. При ответе учитывай историю. Кратко отвечай на вопросы. \
Отвечай на вопрос ровно один раз.


История:
{history}


Вопрос:
{query}


Ответ: """

banking_template = """ Ты - сотрудник банка, который консультирует клиента о банковских продуктах и сервисах. \
Вежливо и кратко отвечай на вопросы клиента. \
Предлагай банковские продукты, которые помогут удовлетворить потребности клиента наилучшим образом. \
При ответе на вопросы опирайся на контекст ниже.


Контекст:
{context}


История:
{history}


Вопрос:
{query}


Ответ: """

# runnable_banking_template = ChatPromptTemplate.from_template(banking_template)

embeddings = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
chat_model = LlamaCpp(
    # llama 13b saiga: -- '../models/model-q4_K(2).gguf'
    # roleplay - mixtral moe 8x7b: -- mixtral-8x7b-moe-rp-story.Q4_K_M
    # mixtral-8x7b-v0.1.Q4_K_M
    model_path='../models/model-q4_K(2).gguf',
    n_gpu_layers=28,  # 28 for llama2 13b, 10 for mixtral
    max_tokens=2000,
    n_batch=256,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    temperature=0.0,
)

prompt_templates = [physics_template, math_template, banking_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

def prompt_router(input):
    # logger.info(f'input:: {input}')  # assert has keys
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # most_similar = prompt_templates[2]  # DEBUG - ONLY FINANCIAL ASSISTANT
    logger.debug(f'most similar:: {most_similar} -- type: {type(most_similar)}')
    logger.info(f'similarity: {similarity, similarity.argmax()}')
    # add context only if similarity.argmax() >= 2 <=> banking template, so we must do RAG
    # logger.critical(f'PromptTemplate -- {PromptTemplate.from_template(most_similar)}, type: {type(PromptTemplate.from_template(most_similar))}')
    return PromptTemplate.from_template(most_similar)
    
store = {}
def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]

def func_logger(args):
    logger.debug(f'{args}')
    return args

# works
query = itemgetter("query") 
context = itemgetter("query") | dummy_retriever | format_docs
query_and_context = RunnablePassthrough.assign(context=context).assign(query=query)
runnable = query_and_context | RunnableLambda(prompt_router) | RunnableLambda(func_logger) | chat_model  #



with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

# print(
#     with_message_history.invoke(
#     input={'query': 'Что такое полярные координаты с точки зрения математики?'}, 
#     config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, "callbacks":[langfuse_callback_handler]}
#     )
# )


print(
    with_message_history.invoke(
        input={'query': ' Что такое физика?'}, 
        config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, "callbacks":[langfuse_callback_handler]}
    )
)

# print(
#     with_message_history.invoke(
#         input={'query': ' Что такое математика?'}, 
#         config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, "callbacks":[langfuse_callback_handler]}
#     )
# )

#########################################################################################

# print(
#     with_message_history.invoke(
#         input={'query': 'Что такое банк?'}, 
#         config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, "callbacks":[langfuse_callback_handler]}
#     )
# )

##########################################################################################

# print(
#     with_message_history.invoke(
#         input={'query': 'Что такое банк?'}, 
#         config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, "callbacks":[langfuse_callback_handler]}
#     )
# )

# ##########################################################################################

# print(
#     with_message_history.invoke(
#         {'query': 'Что ты такое?'}, 
#         config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, "callbacks":[langfuse_callback_handler]}
#     )
# )

# print(
#     with_message_history.invoke(
#         {'query': 'Кто тебя создал?'}, 
#         config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, "callbacks":[langfuse_callback_handler]}
#     )
# )

# print(
#     with_message_history.invoke(
#         {'query': 'Где ты работаешь?'}, 
#         config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, "callbacks":[langfuse_callback_handler]}
#     )
# )

# print(
#     with_message_history.invoke(
#         {'query': 'В каком банке ты работаешь?'}, 
#         config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, "callbacks":[langfuse_callback_handler]}
#     )
# )

# print(
#     with_message_history.invoke(
#         {'query': 'В какой стране ты находишься?'}, 
#         config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, "callbacks":[langfuse_callback_handler]}
#     )
# )

print(
    with_message_history.invoke(
        {'query': 'Что такое математика?'}, 
        config={
            'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, 
            "callbacks":[langfuse_callback_handler]
        }
    )
)

print(
    with_message_history.invoke(
        {'query': 'Какой вопрос я задал только что?'}, 
        config={
            'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, 
            "callbacks":[langfuse_callback_handler]
        }
    )
)

print(
    with_message_history.invoke(
        {'query': 'Повтори мой первый вопрос'}, 
        config={
            'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'}, 
            "callbacks":[langfuse_callback_handler]
        }
    )
)

print('success'.upper())
