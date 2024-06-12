from typing import Any
import sys
sys.path.append('/home/amstel/llm')
sys.path.append('/home/amstel/llm/src')
import pickle
sys.path.append('/')
import numpy as np
from loguru import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.rag_config import N_EMBEDDING_RESULTS, SEPARATOR, TO_REPLACE_SEPARATOR, REPLACE_SEPARATOR_WITH, EMBEDDING_MODEL_NAME, N_RERANK_RESULTS, USE_RERANKER, RERANKING_MODEL, ELBOW_EMBEDDING, ELBOW_RERANKING, MOST_RELEVANT_AT_THE_TOP
from langchain_community.llms import Ollama
from langchain.utils.math import cosine_similarity
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from general_llm.langchain_llama_cpp_api_warpper import LlamaCppApiWrapper
from scenarios.base import BaseScenario
# from langchain_milvus import MilvusCollectionHybridSearchRetriever
from rag.utils import ExtendedMilvusCollectionHybridSearchRetriever as MilvusCollectionHybridSearchRetriever
from rag.utils import BGEDocumentCompressor

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    RRFRanker,
    connections,
)

# def cosine_similarity(a,b):
#     return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def format_docs(docs):
    if TO_REPLACE_SEPARATOR:
        return "\n\n".join(doc.page_content.replace(SEPARATOR, REPLACE_SEPARATOR_WITH) for doc in docs)
    return "\n\n".join(doc.page_content.replace("passage: ", "") for doc in docs)

def def_debugger(inp):
    logger.info(inp)
    return inp


def min_max_scaling(values, min_=None, max_=None):
    if min_ is None:
        min_ = min(values)
    if max_ is None:
        max_ = max(values)
    values = np.array(values)
    values = (values - min_) / (max_ - min_)
    return values


class SberbankConsultant(BaseScenario):
    def __init__(self,):
        self.rag_collections = [
            'full_bge_credits',
            'full_bge_deposits',
            'full_bge_cards',
            'full_bge_other',
        ]
        self.dense_embedding_model = bge_m3_ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,  # Specify the model name
            model_kwargs={'device': 'cpu',}
        )

        self.sparse_model_2_rag_collections = {}
        for rag_collection in self.rag_collections:
            with open(f'/home/amstel/llm/src/rag/sparse_embedding_model_{rag_collection.replace("full_", "")}.pkl', 'rb') as f:
                model = pickle.load(f)
            self.sparse_model_2_rag_collections[rag_collection] = model


    def retriever_router(self, input: str):
        # input: str
        logger.debug(input)
        options = [
            "кредит овердрафт рефинансирование долг",  # assert credit
            "депозит вклад сбережения накопления pay", # assert deposit
            "карта платежная дебетовая манибэк money-back кэшбэк cash-back сберкарта",  # # assert card
            "страховки подписка прайм prime сбол банковские продукты услуги",   # assert other
        ]
        prompt_embeddings = self.dense_embedding_model.embed_documents(options)
        dense_query_embedding = self.dense_embedding_model.embed_query(input.lower())
        # todo: change cosine similarity to bm25
        sparse_similarity = {}
        for sparse_embedding_model_name, sparse_embedding_model in self.sparse_model_2_rag_collections.items():
            # assert order: credit, deposit, card, other
            sparse_similarity[sparse_embedding_model_name] = sum(sparse_embedding_model.embed_query(input.lower()).values())
        dense_similarity = cosine_similarity([dense_query_embedding], prompt_embeddings)[0]

        dense_similarity = min_max_scaling(dense_similarity, min_=0, max_=1)
        sparse_similarity = min_max_scaling(list(sparse_similarity.values()), min_=0.01, max_=10)

        combined_similarity_sum = (dense_similarity * 0.95) + (sparse_similarity * 0.05)
        combined_similarity_product = dense_similarity * sparse_similarity

        logger.info(f'dense similarities: {dense_similarity}')
        logger.info(f'sparse similarities: {sparse_similarity}')

        logger.info(f'combined_similarity_sum: {combined_similarity_sum}')
        logger.info(f'combined_similarity_product: {combined_similarity_product}')

        chosen_rag_collection = self.rag_collections[combined_similarity_sum.argmax()]
        # chosen_rag_collection = self.rag_collections[dense_similarity.argmax()]

        self.sparse_embedding_model = self.sparse_model_2_rag_collections[chosen_rag_collection]

        sparse_search_params = {"metric_type": "IP"}
        dense_search_params = {"metric_type": "IP", "params": {}}

        CONNECTION_URI = "http://localhost:19530"
        connections.connect(uri=CONNECTION_URI)

        pk_field = "doc_id"
        dense_field = "dense_vector"
        sparse_field = "sparse_vector"
        text_field = "text"
        fields = [
            FieldSchema(
                name=pk_field,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                max_length=100,
            ),
            FieldSchema(name=dense_field, dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=65_535),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)


        collection = Collection(name=chosen_rag_collection)
        if not self.sparse_embedding_model.embed_query(input):
            # empty bm25
            retriever = MilvusCollectionHybridSearchRetriever(
                collection=collection,
                rerank=WeightedRanker(1.0),
                anns_fields=['dense_vector'],
                field_embeddings=[self.dense_embedding_model],
                field_search_params=[dense_search_params],
                top_k=N_EMBEDDING_RESULTS,
                text_field="text",
                use_elbow_for_embedding=ELBOW_EMBEDDING,
            )
        else:
            retriever = MilvusCollectionHybridSearchRetriever(
                collection=collection,
                rerank=WeightedRanker(0.5, 0.5),
                # rerank=RRFRanker(k=1),
                anns_fields=['dense_vector', 'sparse_vector'],
                field_embeddings=[self.dense_embedding_model, self.sparse_embedding_model],
                field_search_params=[dense_search_params, sparse_search_params],
                top_k=N_EMBEDDING_RESULTS,
                text_field="text",
                use_elbow_for_embedding=ELBOW_EMBEDDING,
            )

        if USE_RERANKER:
            compressor = BGEDocumentCompressor(
                top_n=N_RERANK_RESULTS,
                model_name_or_path=RERANKING_MODEL,
                elbow=ELBOW_RERANKING,
                most_relevant_at_the_top=MOST_RELEVANT_AT_THE_TOP,
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            return compression_retriever
        return retriever

    def handle(self, user_query: Any, chat_history: Any = [], context: Any = {}):
        #todo: chat_history
        # is not used
        retriever = RunnableLambda(self.retriever_router)

        llama_raw_template_system = """<|start_header_id|>system<|end_header_id|>\nТы - приветливый ИИ, разработанный в Сбер Банке (Беларусь). Ты знаешь только русский язык. Основываясь на контексте ниже, правдиво и полно отвечай на вопросы.<|eot_id|>"""
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
    # q = "условия по СберКарте"
    # q = "безотзывный депозит в белорусских рублях (BYN) сохраняй, какие ставки?"
    # q = "Какие есть карты для физических лиц?"
    # q = "как накопить ребенку на образование"
    # q = "Сравни карты"
    # q = "Сравни карты по выгоде"
    # q = "Какой кредит самый выгодный?"
    # q = "Самый выгодный процент по кредиту на авто"
    # q = "Купи авто в Online условия"
    # q = "Собираюсь в отпуск в Турцию. Подбери мне страховку."
    # q = "Подбери мне страховку"
    # q = "Подбери мне карту"
    # q = "максимальные ставки по безотзывным депозитам в белорусских рублях таблица"
    # q = "Подбери мне кредит"
    # q = "Подбери мне депозит"
    q = "Какой депозит самый выгодный?"

    consultant = SberbankConsultant()
    response, context = consultant.handle(user_query=q)
    logger.info(f"response: {response}")
    logger.info(f"context: {context}")



    # Question

    # logger.warning(q)
    # response = rag_chain.invoke(q)

    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    #
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    #
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    #
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    #
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    # #
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")
    #
    # response = rag_chain.invoke()
    # logger.info(f"response: {response}")
    # logger.warning(q)
    # response = rag_chain.invoke(q)
    # logger.info(f"response: {response}")

    #
    # response = rag_chain.invoke()
    # logger.info(f"response: {response}")
    #
    # response = rag_chain.invoke()
    # logger.info(f"response: {response}")
    #
    # response = rag_chain.invoke()
    # logger.info(f"response: {response}")
    #
    # response = rag_chain.invoke()
    # logger.info(f"response: {response}")
    #
    # response = rag_chain.invoke()
    # logger.info(f"response: {response}")

