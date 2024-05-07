import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
from rag_config import SPLITTER_SEPARATORS, CHUNK_OVERLAP, CHUNK_SIZE, RAG_COLLECTION_NAME
from langchain_experimental.text_splitter import SemanticChunker


if __name__ == '__main__':
    with open('/home/amstel/llm/src/web_scraping/bank_scraper/docs_final.pkl', 'rb') as f:
        docs = pickle.load(f)
    docs = [doc for doc in docs if doc.page_content != 'Извините, информация не найдена.']
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SPLITTER_SEPARATORS,
        keep_separator=False,
    )

    # "percentile", "standard_deviation", "interquartile"
    # text_splitter = SemanticChunker(embedding_model,
    #                                 breakpoint_threshold_type='interquartile')

    logger.info(type(docs[-1]))
    splits = text_splitter.split_documents(docs)
    logger.info(type(splits[-1]))
    # run once
    db = Milvus.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name=RAG_COLLECTION_NAME,
    )