import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from MIlvusHybrid import MilvusHybrid
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
from rag_config import RAG_COLLECTION_NAME
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from rag_config import RAG_COLLECTION_NAME, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, SPLITTER_SEPARATORS, SEPARATOR
from utils import MarkdownTextSplitter, BM25
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
)


def main(slug: str):
    with open(f'rag_w_summary_results_{slug}.pkl', 'rb') as f:
        results = pickle.load(f)

    dense_embedding_model = HuggingFaceEmbeddings(
            model_name='BAAI/bge-m3',  # Specify the model name
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
    new_corpus = [x.page_content for x in new_documents]

    sparse_embedding_model = BM25SparseEmbedding(
        corpus=new_corpus,
        language='ru',
    )  # it is fitted at initialization

    with open(f'sparse_embedding_model_bge_{slug}.pkl', 'wb') as f:
        pickle.dump(sparse_embedding_model, f)


    # to add source?
    # milvus insert
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
    collection = Collection(
        name='full_bge_'+slug, schema=schema, consistency_level="Strong"
    )
    dense_index = {"index_type": "FLAT", "metric_type": "IP"}
    collection.create_index("dense_vector", dense_index)
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_vector", sparse_index)
    collection.flush()

    print('start embedding')
    entities = []
    for text in new_corpus:
        entity = {
            dense_field: dense_embedding_model.embed_documents([text])[0],
            sparse_field: sparse_embedding_model.embed_query(text),
            text_field: text,
        }
        entities.append(entity)
    collection.insert(entities)
    collection.load()


if __name__ == '__main__':
    # for slug in [
    #     'other',
    #     'cards',
    #     'credits',
    #     'deposits'
    # ]:
        # main(slug)

    pkl_input_filepaths = [
        'rag_w_summary_results_other.pkl',
        'rag_w_summary_results_cards.pkl',
        'rag_w_summary_results_credits.pkl',
        'rag_w_summary_results_deposits.pkl',
    ]
    bm = BM25()
    bm.load_data_many(pkl_input_filepaths=pkl_input_filepaths)
    bm.fit()
    bm.save_bse_model(filepath='router_bm25_model.pkl')
