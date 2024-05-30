# тест на наличие данных. если в результате суммраизацции преимущества или недостатки = пусто, то их векторного представления тоже не будет

import sys
sys.path.append('/home/amstel/llm')
from src.mongodb.utils import MongoConnector
from pymilvus import MilvusClient
from loguru import logger


milvus_client = MilvusClient(uri="http://localhost:19530")
con_product_reviews = MongoConnector(operation='read',
                                         db_name='scraped_data',
                                         collection_name='product_review_summarizations'
                                         )
cursor_product_reviews = list(con_product_reviews.read_many({}))
logger.warning(f'len: {len(cursor_product_reviews)}')

for text_entity in ("advantages", "disadvantages"):
    assert text_entity in ("advantages", "disadvantages")
    collection_name = f'summarize_product_reviews_embeddings_{text_entity}'
    for i, review in enumerate(cursor_product_reviews):
        product_name = review.get('product_name')
        res = milvus_client.query(
            collection_name=collection_name,
            filter=f"product_name in ['{product_name}']",
        )
        try:
            assert res[0].get('product_name') == product_name
        except IndexError:
            logger.critical(product_name, res)
    # logger.warning(res)
