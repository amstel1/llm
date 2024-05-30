import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
from rag_config import RAG_COLLECTION_NAME
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from rag_config import RAG_COLLECTION_NAME, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, SPLITTER_SEPARATORS
from custom_splitter import MarkdownTextSplitter

if __name__ == '__main__':
    # with open('/home/amstel/llm/src/web_scraping/bank_scraper/docs_final.pkl', 'rb') as f:
    #     docs = pickle.load(f)
    with open('rag_w_summary_results_other.pkl', 'rb') as f:
        results = pickle.load(f)


    embedding_model = HuggingFaceEmbeddings(
        # model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
        model_name=EMBEDDING_MODEL_NAME,  # "intfloat/multilingual-e5-large"
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
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
            for frag in text_splitter.split_text(formatted):
                new_documents.append(
                    Document(
                        page_content=summarized + ' || ' + frag,
                        # page_content=formatted,
                        metadata=metadata
                    )
                )
    print("new_documents", len(new_documents))
    # "percentile", "standard_deviation", "interquartile"
    # text_splitter = SemanticChunker(embedding_model,
    #                                 breakpoint_threshold_type='interquartile')

    # logger.info(type(docs[-1]))
    # splits = text_splitter.split_documents(docs)
    # logger.info(type(splits[-1]))

    # run once
    db = Milvus.from_documents(
        documents=new_documents,
        embedding=embedding_model,
        collection_name=RAG_COLLECTION_NAME,
    )