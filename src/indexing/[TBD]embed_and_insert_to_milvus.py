
import sys
sys.path.append('/home/amstel/llm')
from src.mongodb.mongo_utils import MongoConnector
import numpy as np
from langchain_community.embeddings import LlamaCppEmbeddings
from loguru import logger
from datetime import datetime
from pymilvus import MilvusClient
from langchain_community.vectorstores import Milvus
from langchain_community.retrievers import MilvusRetriever

class Embedder:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def unite_embed(self, input: list[str], separator: str = " ") -> list[float]:
        """unite -> embed"""
        united_string = separator.join(input)
        embedding = self.embedding_model.embed_query(united_string)
        return embedding

    def embed_unite(self, input: list[str], strategy='average') -> list[float]:
        embeddings = np.array(self.embedding_model.embed_documents(input))
        if strategy == 'average':
            embedding = np.array(embeddings).mean(axis=0)
        return embedding

def insert_to_vectordb(collection_name: str, embedding: float, text_entity, product_name: str, product_type: str = "Стиральная машина") -> None:
    """
    Inserts

    :param collection: collection_name
    :param embedding: list[float] - supported only one embedding currently
    :param text_entity: str, must be in ("advantages", "disadvantages") - defines prefix for the embedding field
    """
    review_milvus_data = {
        "insertion_datetime": str(datetime.now()),
        "product_type": product_type,
        "product_name": product_name,
        "embedding_model": "nomic-embed-text-v1.5.Q8_0",
        f"{text_entity}_embeddings": emb_2_normalized,
    }
    # data.append(review_milvus_data)

    # write one by one to check for duplicates - todo: rewrite for batches
    # Search if this embedding already exists to prevent duplicates
    search_params = {"metric_type": "L2", "params": {"product_name": review.get('product_name')}}

    results = milvus_client.query(
        collection_name=collection_name,
        filter=f"product_name in ['{product_name}']",
    )

    if results:
        if results[0]:
            try:
                assert results[0].get('product_name') == product_name
            except IndexError:
                logger.critical("проверка не пройдена - хз что делать")
        else:
            milvus_client.insert(collection_name=collection_name, data=[review_milvus_data])
            logger.info(f'inserted: {review_milvus_data.get("product_name")}')
    else:
        milvus_client.insert(collection_name=collection_name, data=[review_milvus_data])
        logger.warning(f'inserted: {review_milvus_data.get("product_name")}')

if __name__ == '__main__':
    con_product_reviews = MongoConnector(operation='read',
                                         db_name='scraped_data',
                                         collection_name='product_review_summarizations'
                                         )
    cursor_product_reviews = list(con_product_reviews.read_many({}))
    logger.warning(f'len: {len(cursor_product_reviews)}')
    embedding_model = LlamaCppEmbeddings(
        model_path='/home/amstel/llm/models/Publisher/Repository/nomic-embed-text-v1.5.Q8_0.gguf',
        n_gpu_layers=13,
        n_batch=64,
        verbose=False,
    )
    embedder = Embedder(embedding_model=embedding_model)
    milvus_client = MilvusClient(uri="http://localhost:19530")

    for text_entity in ("advantages", "disadvantages"):
        assert text_entity in ("advantages", "disadvantages")
        collection_name = f'summarize_product_reviews_embeddings_{text_entity}'
        for i, review in enumerate(cursor_product_reviews):
            text = review.get(text_entity)  # either advantages or disadvantages
            if not text:
                continue

            # the hidden space collapses - tested 2024-05-02, not recommended
            # emb_1 = embedder.embed_unite(text, strategy='average')
            # emb_1_normalized = emb_1 / np.linalg.norm(emb_1)
            # emb_1_normalized = list(emb_1_normalized)

            # the hidden space does not collapse
            emb_2_normalized = embedder.unite_embed(text, separator=" ")
            # this will break if the embedding model returns unnormalized values
            assert np.linalg.norm(emb_2_normalized) >=0.99
            # emb_2_normalized = emb_2 / np.linalg.norm(emb_2)

            insert_to_vectordb(
                collection_name=collection_name,
                embedding=emb_2_normalized,
                text_entity=text_entity,
                product_name=review.get('product_name'),
                product_type="Стиральная машина",  # todo: get this from mongodb
            )

    logger.info('success')