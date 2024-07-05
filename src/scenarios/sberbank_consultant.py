from typing import Any, Optional, List, Dict
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
from scenarios.scenario_router import ScenarioRouter, Route
from scenarios.shopping_assistant import chat_history_list_to_str
# from langchain_milvus import MilvusCollectionHybridSearchRetriever
from rag.utils import ExtendedMilvusCollectionHybridSearchRetriever as MilvusCollectionHybridSearchRetriever
from rag.utils import BGEDocumentCompressor
from langchain_core.output_parsers import JsonOutputParser
from general_llm.llm_endpoint import call_generation_api, MODEL_NAME, call_generate_from_query_api
from general_llm.prompt_construction import Llama3PromptTemplate, Gemma2PromptTemplate
import json

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

class RetrieverRouter(ScenarioRouter):
    # define only init
    # call with proper grammar
    def __init__(self):
        self.system_prompt = 'You are a state-of-the-art intent classifer.'
        self.user_prompt_with_chat_history_placeholder = """Based on the user input and the chat history, identify which route the user's input most closely relates to. Your decision should take into account the context provided by the chat history. Respond with the most relevant route name from the given mapping.

route mapping:
full_bge_credits: кредит овердрафт рефинансирование долг
full_bge_deposits: депозит вклад сбережения накопления pay
full_bge_cards: карта платежная дебетовая манибэк money-back кэшбэк cash-back сберкарта
full_bge_other: страховки подписка прайм prime сбол банковские продукты услуги

chat history:
{chat_history}

user_input:
{user_input}

Please respond with the most relevant route name as JSON."""
        self.user_prompt_without_chat_history_placeholder = """Based on the user input identify which route the user's input most closely relates to. Respond with the most relevant route name from the given mapping.

route mapping:
full_bge_credits: кредит овердрафт рефинансирование долг
full_bge_deposits: депозит вклад сбережения накопления pay
full_bge_cards: карта платежная дебетовая манибэк money-back кэшбэк cash-back сберкарта
full_bge_other: страховки подписка прайм prime сбол банковские продукты услуги

user input:
{user_input}"""
        self.assistant_starts_with = "\nJSON:"

    def route(self,
              user_query: str,
              chat_history: Optional[List[Dict[str, str]]] = None,
              grammar_path: str = None,  # force json output
              stop: list[str] = [],
              ) -> Route:
        if (chat_history is None) or (not chat_history) or (len(chat_history)==0):
            generation_result = call_generate_from_query_api(
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt_without_chat_history_placeholder.format(**{"user_input": user_query,}),
                assistant_must_start_with=self.assistant_starts_with,
                grammar_path=grammar_path,
                stop=stop,
            )
        else:
            # done: chat_history must be parsed, already implemented at .shopping_assistant, reuse
            str_chat_history = chat_history_list_to_str(chat_history)
            generation_result = call_generate_from_query_api(
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt_with_chat_history_placeholder.format(
                    **{"user_input": user_query, "chat_history": str_chat_history,}),
                assistant_must_start_with=self.assistant_starts_with,
                grammar_path=grammar_path,
                stop=stop,
            )
        logger.debug(f'0407 - router - {generation_result}')
        selected_route_dict = JsonOutputParser().parse(generation_result)
        # selected_route_dict = json.loads(generation_result)
        selected_route = selected_route_dict.get('route')
        logger.warning(selected_route)
        if isinstance(selected_route, list):
            assert len(selected_route) == 1
            selected_route_str = selected_route[0]
        elif isinstance(selected_route, str):
            selected_route_str = selected_route
        return selected_route_str

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


    def retriever_router(self, input: str, chat_history: list):
        logger.debug(input)

        retriever_router = RetrieverRouter()
        chosen_rag_collection = retriever_router.route(
            user_query=input,
            chat_history=chat_history,
            grammar_path='/home/amstel/llm/src/grammars/sberbank_consultant_router.gbnf',  # force json output
        )
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
        retriever = self.retriever_router(input=user_query, chat_history=chat_history)

        # edit 2806 - for citations
        system_promt = "Ты - сотрудник Сбер Банка (Беларусь). Ты знаешь только русский язык. Основываясь на контексте ниже, правдиво и полно отвечай на вопросы. "

        user_prompt_placeholder = """история разговора: {chat_history_str}\nконтекст:{context}\n\nВопрос:{question}\n
Ответь на вопрос выше полно, правдиво и развернуто. """

        chat_history_str = chat_history_list_to_str(chat_history)

        retrieved_docs = retriever.invoke(user_query)
        id_2_doc_id_source = {}
        filtered_page_contents = []
        for i, doc in enumerate(retrieved_docs) :
            id_2_doc_id_source[i] = {
                'doc_id': doc.metadata['doc_id'],
                'source': doc.metadata['source'],
            }
            if TO_REPLACE_SEPARATOR:
                page_content = f'id: {i}\n{doc.page_content.replace(SEPARATOR, REPLACE_SEPARATOR_WITH).strip()}'
                filtered_page_contents.append(page_content)
            else:
                page_content = f'id: {i}\n{doc.page_content.strip()}'
                filtered_page_contents.append(page_content)

        retrieved_docs_str = "\n\n".join(filtered_page_contents)
        user_prompt = user_prompt_placeholder.format(
            context=retrieved_docs_str,
            question=user_query,
            chat_history_str=chat_history_str,
        )

        # assistant_starts_with = '\nJSON:'  # remove citations
        if 'llama' in MODEL_NAME: prompt_template = Llama3PromptTemplate
        if 'gemma' in MODEL_NAME: prompt_template = Gemma2PromptTemplate
        prompt = prompt_template().create_prompt_from_user_query(
            system_prompt_clean=system_promt,
            user_query=user_prompt,
            # assistant_must_start_with=assistant_starts_with,
        )

        llm_response = call_generation_api(prompt=prompt, grammar_path='/home/amstel/llm/src/grammars/sberbank_rag_citations.gbnf')
        logger.info(f'sberbank consultant response type 1 - {type(llm_response)}')
        logger.info(f'sberbank consultant response - {llm_response}')
        # llm_response = JsonOutputParser().parse(llm_response)
        # llm_response = json.loads(llm_response)  # remove citations, no dict anymore
        logger.info(f'sberbank consultant response type 2 - {type(llm_response)}')
        assert isinstance(llm_response, str)
        response = llm_response
        # response_keys = list(llm_response.keys())
        # answer_key = [x for x in response_keys if 'answer' in x.lower()][0]
        # ids_key = [x for x in response_keys if 'id' in x.lower()][0]
        context['current_step'] = 'sberbank_consultant'
        if 'previous_steps' not in context: context['previous_steps'] = []
        context['previous_steps'].append('sberbank_consultant')
        context['scenario_name'] = "just_chatting"   # ? why
        # context['citations_lookup'] = id_2_doc_id_source
        # context['cited_sources'] = llm_response.get(ids_key)
        # response = llm_response.get(answer_key)
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
    # q = "Какие есть кредиты на авто?"
    q = "какие есть кредиты для мобильного телефона"
    consultant = SberbankConsultant()
    response, context = consultant.handle(user_query=q)
    logger.info(f"response: {response}")
    logger.info(f"context: {context}")



