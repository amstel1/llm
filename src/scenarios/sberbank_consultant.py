from typing import Any
import sys
sys.path.append('/home/amstel/llm')
sys.path.append('/home/amstel/llm/src')

sys.path.append('/')
import numpy as np
from loguru import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.rag_config import N_NEIGHBORS, SEPARATOR, TO_REPLACE_SEPARATOR, REPLACE_SEPARATOR_WITH, EMBEDDING_MODEL_NAME, N_RERANK_RESULTS, USE_RERANKER, RERANKING_MODEL
from langchain_community.llms import Ollama
from langchain.utils.math import cosine_similarity
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from general_llm.langchain_llama_cpp_api_warpper import LlamaCppApiWrapper
from scenarios.base import BaseScenario

# def cosine_similarity(a,b):
#     return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def format_docs(docs):
    if TO_REPLACE_SEPARATOR:
        return "\n\n".join(doc.page_content.replace(SEPARATOR, REPLACE_SEPARATOR_WITH) for doc in docs)
    return "\n\n".join(doc.page_content.replace("passage: ", "") for doc in docs)

def def_debugger(inp):
    logger.info(inp)
    return inp



class SberbankConsultant(BaseScenario):
    def __init__(self,):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def retriever_router(self, input: str):
        # input: str
        logger.debug(input)
        options = [
            "кредит овердрафт рефинансирование долг",
            "депозит вклад сбережения накопления pay",
            "карта платежная дебетовая манэбэк money-back кэшбэк cash-back",
            "страховки подписка cберпрайм sberprime сбол банковские продукты услуги условия обслуживания",
        ]
        rag_collections = [
            'credits_400_0',
            'deposits_400_0',
            'cards_400_0',
            'other_400_0',
        ]
        prompt_embeddings = self.embedding_model.embed_documents(options)
        query_embedding = self.embedding_model.embed_query(input.lower())
        similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
        logger.info(f'similarities: {similarity}')
        chosen_rag_collection = rag_collections[similarity.argmax()]

        db = Milvus(embedding_function=self.embedding_model, collection_name=chosen_rag_collection, )
        retriever = db.as_retriever(search_type='mmr', search_kwargs={"k": N_NEIGHBORS, })

        if USE_RERANKER:
            # DEFAULT:  ms-marco-MultiBERT-L-12 - лучшая
            # ce-esci-MiniLM-L12-v2 - плоховато
            # ms-marco-MiniLM-L-12-v2 -- дерьмо какое то
            # rank-T5-flan - kind of okay
            # try: doc2query/msmarco-russian-mt5-base-v1 - весит 2+ gb
            compressor = FlashrankRerank(top_n=N_RERANK_RESULTS,
                                         model=RERANKING_MODEL)  # doc2query/msmarco-russian-mt5-base-v1
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            return compression_retriever
        return retriever

    def handle(self, user_query: Any, chat_history: Any = [], context: Any = {}):
        #todo: chat_history is not used
        retriever = RunnableLambda(self.retriever_router)

        llama_raw_template_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nТы - приветливый ИИ, разработанный в Сбер Банке (Беларусь). Ты знаешь только русский язык. Основываясь на контексте ниже, правдиво и полно отвечай на вопросы.<|eot_id|>"""
        llama_raw_template_user = """<|start_header_id|>user<|end_header_id|>\nКонтекст:\n\n{context}\n\nВопрос:\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        # Prompt
        debugger = RunnablePassthrough(def_debugger)
        prompt = PromptTemplate.from_template(template=llama_raw_template_system + llama_raw_template_user)

        llm = LlamaCppApiWrapper()
        # Chain
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | def_debugger
                | llm
                | StrOutputParser()
        )
        response = rag_chain.invoke(user_query)
        context['current_step'] = 'sberbank_consultant'
        if 'previous_steps' not in context: context['previous_steps'] = []
        context['previous_steps'].append('sberbank_consultant')
        context['scenario_name'] = "just_chatting"
        return response, context

if __name__ == '__main__':
    q = "Какие есть кредиты для физических лиц?"

    consultant = SberbankConsultant()
    response, context = consultant.handle(user_query=q)
    logger.info(f"response: {response}")
    logger.info(f"context: {context}")



    # Question

    # logger.warning(q)
    # response = rag_chain.invoke(q)

    #
    # q = "безотзывный депозит в белорусских рублях сохраняй, какие ставки?"
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    #
    # q = "Какие есть карты для физических лиц?"
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    #
    # q = "Какие есть карты для физических лиц?"
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    #
    # q = "как накопить ребенку на образование"
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    #
    # q = "Сравни карты"
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    # #
    # q = "Какой кредит самый выгодный?"
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    #
    # response = rag_chain.invoke("Какой депозит самый выгодный?")
    # logger.info(f"response: {response}")
    # q = "Самый выгодный процент по кредиту на авто"
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")

    #
    # response = rag_chain.invoke("Подбери мне кредит")
    # logger.info(f"response: {response}")
    #
    # response = rag_chain.invoke("Подбери мне депозит")
    # logger.info(f"response: {response}")
    #
    # response = rag_chain.invoke("Подбери мне карту")
    # logger.info(f"response: {response}")
    #
    # response = rag_chain.invoke("Подбери мне страховку")
    # logger.info(f"response: {response}")
    #
    # response = rag_chain.invoke("Собираюсь в отпуск в Турцию. Подбери мне страховку.")
    # logger.info(f"response: {response}")

