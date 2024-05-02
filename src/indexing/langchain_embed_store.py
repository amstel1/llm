# todo: prevent duplicates
from langchain_community.vectorstores import Milvus
from langchain_core.embeddings import Embeddings
from typing import List
import sys
sys.path.append('/home/amstel/llm')
from src.mongodb.mongo_utils import MongoConnector
from datetime import datetime
from langchain_community.embeddings import LlamaCppEmbeddings
from loguru import logger


if __name__ == '__main__':
    separator = " | "
    embedding_model = LlamaCppEmbeddings(
        model_path='/home/amstel/llm/models/Publisher/Repository/nomic-embed-text-v1.5.Q8_0.gguf',
        n_gpu_layers=13,
        n_batch=64,
        verbose=False,
    )
    con_product_reviews = MongoConnector(operation='read',
                                         db_name='scraped_data',
                                         collection_name='product_review_summarizations'
                                         )
    cursor_product_reviews = list(con_product_reviews.read_many({}))

    for gist_entity in ('advantages', 'disadvantages'):
        texts = []
        metadatas = []
        embeddings = []
        for review in cursor_product_reviews:
            page_content_list = review.pop(gist_entity)
            if isinstance(page_content_list, list):
                page_content = separator.join(page_content_list)
            else:
                page_content = ''
            metadata = {}
            metadata['product_name'] = review.get('product_name')
            metadata['text_entity'] = gist_entity
            metadata['product_type'] = "Стиральная машина"
            metadata['insertion_datetime'] = str(datetime.now())
            metadata['embedding_model'] = "nomic-embed-text-v1.5.Q8_0"
            logger.info(page_content)
            metadatas.append(metadata)
            texts.append(page_content)
            embedding = embedding_model.embed_query(page_content)
            embeddings.append(embedding)


        milvus_vector_store = Milvus.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embedding_model,
            collection_name=f"summarize_product_reviews_embeddings_{gist_entity}",
        )
        logger.warning('success')