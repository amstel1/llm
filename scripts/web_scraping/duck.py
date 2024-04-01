enable_langfuse = True

from datetime import datetime
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
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from operator import itemgetter
import os


if enable_langfuse:
    trace_name = f'rp-moe-deposit_{str(datetime.now())}'
    os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-a45f2a3d-4085-4170-b337-8cc2f1921aef"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-26acbd19-74af-4cb4-93e1-919526c13921"


    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler

    langfuse = Langfuse()
    langfuse_callback_handler = CallbackHandler(trace_name=trace_name)
    langfuse_callback_handler = CallbackHandler(
        secret_key="sk-lf-26acbd19-74af-4cb4-93e1-919526c13921",
        public_key="pk-lf-a45f2a3d-4085-4170-b337-8cc2f1921aef",
        host="http://localhost:3000",
    )
    assert langfuse_callback_handler.auth_check()

search = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="d", max_results=20)
searching_tool = DuckDuckGoSearchResults(api_wrapper=search,)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """<s> [INST] Ты - помощник в выборе товаров. Ты обобщаешь и суммаризуешь информацию по запросу. Контекст содержит результаты поиска в интернете.

Результаты поиска:
{context}

Запрос:
{query} [/INST] </s>

Твой ответ: """


chat_model = LlamaCpp(
    # llama 13b saiga: -- '../models/model-q4_K(2).gguf'
    # roleplay - mixtral moe 8x7b: -- mixtral-8x7b-moe-rp-story.Q4_K_M
    # mixtral-8x7b-v0.1.Q4_K_M
    model_path='../models/model-q4_K(2).gguf',
    n_gpu_layers=28,  # 28 for llama2 13b, 10 for mixtral
    max_tokens=500,
    n_batch=128,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    temperature=0.0,
)

store = {}
def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


query = itemgetter("query")
context = query | searching_tool # | format_docs
query_and_context = RunnablePassthrough.assign(context=context).assign(query=query)
runnable = query_and_context | PromptTemplate.from_template(template) | chat_model  # RunnableLambda

google_search_query = 'https://www.google.com/search?q=%D0%BB%D1%83%D1%87%D1%88%D0%B8%D0%B5+%D1%82%D0%B5%D0%BB%D0%B5%D1%84%D0%BE%D0%BD%D1%8B+%D0%B4%D0%BE+1300+%D1%80%D1%83%D0%B1%D0%BB%D0%B5%D0%B9+%D0%BC%D0%B8%D0%BD%D1%81%D0%BA&oq=%D0%BB%D1%83%D1%87%D1%88%D0%B8%D0%B5+%D1%82%D0%B5%D0%BB%D0%B5%D1%84%D0%BE%D0%BD%D1%8B+%D0%B4%D0%BE+1300+%D1%80%D1%83%D0%B1%D0%BB%D0%B5%D0%B9+%D0%BC%D0%B8%D0%BD%D1%81%D0%BA'

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

if enable_langfuse:
    print(
        with_message_history.invoke(
            {'query': 'лучшие телефоны до 1300 рублей минск'},
            config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'},
                    "callbacks": [langfuse_callback_handler]}
        )
    )
else:
    print(
        with_message_history.invoke(
            {'query': 'лучшие телефоны до 1300 рублей минск'},
            config={'configurable': {'user_id': '1-1S8209H', 'conversation_id': 'conv_1'},
                    }
        )
    )

# step 1 - paraphrase query
# step 2 - search

print('success'.upper())
