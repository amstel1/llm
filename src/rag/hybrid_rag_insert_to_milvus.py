import sys
from typing import Dict, Any
import pandas as pd
import numpy as np
sys.path.append('/home/amstel/llm/src')
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from MIlvusHybrid import MilvusHybrid
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from rag.rag_config import RAG_COLLECTION_NAME, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, SPLITTER_SEPARATORS, SEPARATOR
from rag.utils import MarkdownTextSplitter
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
)
from etl_jobs.base import Write, StepNum


class SberbankWebsiteSummaryWrite(Write):
    def __init__(self, products: list[str] = ['cards', 'other', 'deposits']):
        self.products = products

    def write_utility(self, results: dict, slug: str):
        """
        :param results: dict[Link <str>, Article Summary <str>]
        :param slug: Product Name <str>, e.g deposits, cards, credits, other
        :return:
        """
        with open(f'rag_w_summary_results_{slug}.pkl', 'rb') as f:
            results = pickle.load(f)

        dense_embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,  # Specify the model name
                model_kwargs={'device':'cpu',}
                  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
                 # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SPLITTER_SEPARATORS,
            keep_separator=False,
        )
        # text_splitter = MarkdownTextSplitter()
        new_documents = []
        for i, (source_link,r) in enumerate(results.items()):
            metadata = {'source': source_link}
            formatted = r['formatted']
            summarized = r['summarized']
            stop_word_present = False
            for stop_word in ['LaCard', 'ComPass', 'SberDaily', 'Моцная', 'БАТЭ', 'Bonus', "БЕЛКАРТ Pay", "КартаFUN"]:
                if stop_word in formatted:
                    stop_word_present = True
                    break
            if not stop_word_present:
                for _ in [0]: #text_splitter.split_text(formatted):
                    frag = formatted
                    new_documents.append(
                        Document(
                            page_content=summarized + SEPARATOR + frag,
                            # page_content=formatted,
                            metadata=metadata
                        )
                    )
        print("new_documents", len(new_documents))
        new_corpus = [(x.metadata['source'], x.page_content) for x in new_documents]

        sparse_embedding_model = BM25SparseEmbedding(
            corpus=[x[1] for x in new_corpus],
            language='ru',
        )  # it is fitted at initialization

        with open(f'/home/amstel/llm/src/rag/sparse_embedding_model_bge_{slug}.pkl', 'wb') as f:
            pickle.dump(sparse_embedding_model, f)


        # to add source?
        # milvus insert
        CONNECTION_URI = "http://localhost:19530"
        connections.connect(uri=CONNECTION_URI)
        pk_field = "doc_id"
        dense_field = "dense_vector"
        sparse_field = "sparse_vector"
        text_field = "text"
        source_field = "source"
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
            FieldSchema(name=source_field, dtype=DataType.VARCHAR, max_length=2048),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)

        collection = Collection(
            name='full_bge_'+slug, schema=schema, consistency_level="Strong"
        )

        # drop by name here
        # try:
        #     collection.drop()
        # except Exception as e:
        #     logger.error(e)

        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        collection.create_index("dense_vector", dense_index)
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        collection.create_index("sparse_vector", sparse_index)
        collection.flush()

        print('start embedding')
        entities = []
        for doc in new_corpus:
            source_link, text = doc
            entity = {
                dense_field: dense_embedding_model.embed_documents([text])[0],
                sparse_field: sparse_embedding_model.embed_query(text),
                text_field: text,
                source_field: source_link
            }
            entities.append(entity)
        collection.insert(entities)
        collection.load()

    def write(self, data: Dict[StepNum, Any] = None) -> None:
        # all_resutls = data.get('step_0') # dict[product, dict[link_str, summary_str]]
        for product in self.products:
            # results = all_resutls.get

            self.write_utility(results=[], slug=product)

class FewShotQAWrite(Write):
    def __init__(self, schema_names: list[str] = ['tv', 'mobile', 'fridge', 'washing_machine']):
        self.schema_names = schema_names

    def write_utility(self, results: dict, schema_name: str):
        """
        :param results: dict[Link <str>, Article Summary <str>]
        :param slug: Product Name <str>, e.g deposits, cards, credits, other
        :return:
        """
        # read source = excel
        df = pd.read_excel('/home/amstel/llm/src/text2sql/examples.xlsx', sheet_name=schema_name) # cols: user, sql
        df['user'] = df['user'].str.lower()
        dense_embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device':'cpu',}
            )

        corpus = [x.lower() for x in np.concatenate(df['user'].str.split(' '))]
        sparse_embedding_model = BM25SparseEmbedding(
            corpus=corpus,
            language='ru',
        )  # it is fitted at initialization

        with open(f'/home/amstel/llm/src/rag/FewShotQA_{schema_name}.pkl', 'wb') as f:
            pickle.dump(sparse_embedding_model, f)


        CONNECTION_URI = "http://localhost:19530"
        connections.connect(uri=CONNECTION_URI, db_name=schema_name)
        fields = [
            FieldSchema(name="q", dtype=DataType.VARCHAR, is_primary=True, max_length=2048, ),
            FieldSchema(name="q_dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="q_sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="a", dtype=DataType.VARCHAR, max_length=2048),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
        collection = Collection(name='q_a_index', schema=schema, )


        # drop by name here
        # try:
        #     collection.drop()
        # except Exception as e:
        #     logger.error(e)

        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        collection.create_index("q_dense_vector", dense_index)
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        collection.create_index("q_sparse_vector", sparse_index)
        collection.flush()

        print('start embedding')
        entities = []
        for i, row in df.iterrows():
            user_statement_example, sql_reference = row['user'], row['sql']
            entity = {
                "q": user_statement_example ,
                "q_dense_vector": dense_embedding_model.embed_documents([user_statement_example])[0] ,
                "q_sparse_vector": sparse_embedding_model.embed_query(user_statement_example),
                "a": sql_reference
            }
            entities.append(entity)
        collection.insert(entities)
        collection.load()

    def write(self, data: Dict[StepNum, Any] = None) -> None:
        # all_resutls = data.get('step_0') # dict[product, dict[link_str, summary_str]]
        for schema in self.schema_names:
            # results = all_resutls.get
            self.write_utility(results=[], schema_name=schema)


if __name__ == '__main__':
    # for slug in [
    #     'other',
    #     'cards',
    #     'credits',
    #     'deposits'
    # ]:
    #     main(slug)

    # pkl_input_filepaths = [
    #     'rag_w_summary_results_other.pkl',
    #     'rag_w_summary_results_cards.pkl',
    #     'rag_w_summary_results_credits.pkl',
    #     'rag_w_summary_results_deposits.pkl',
    # ]
    # bm = BM25()
    # bm.load_data_many(pkl_input_filepaths=pkl_input_filepaths)
    # bm.fit()
    # bm.save_bse_model(filepath='router_bm25_model.pkl')

    # w = SberbankWebsiteSummaryWrite()
    # w.write(data=None)


    w = FewShotQAWrite(schema_names=[
        'tv',
        'mobile',
        'fridge',
        'washing_machine',
    ])
    w.write(data=None)