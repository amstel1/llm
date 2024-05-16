import sys
import os
sys.path.append('/home/amstel/llm')
from typing import Dict, Tuple, List
from src.mongodb.utils import MongoConnector
import pickle
from loguru import logger
import os





if __name__ == '__main__':
    pass

    # cursor = MongoConnector(operation='write', db_name='scraped_data', collection_name='product_details')
    # cursor.write_many(product_details)
    #
    # cursor = MongoConnector(operation='write', db_name='scraped_data', collection_name='product_reviews')
    # cursor.write_many(reviews_details)
    # logger.info('success')
    # os.remove('/home/amstel/llm/out/QueryDetailsReviews.pkl')
    # logger.warning(len(reviews_details))

    # with open('/home/amstel/llm/out/summarized_reviews.pkl', 'rb') as f:
    #     summarized_reviews = pickle.load(f)
    # logger.info(len(summarized_reviews))
    # cursor = MongoConnector(operation='write', db_name='scraped_data', collection_name='product_review_summarizations')
    # cursor.write_many(summarized_reviews)
    # for k,v in summarized_reviews.items():
    #     try:
    #         cursor.write_one({k:v})
    #     except Exception as e:
    #         logger.critical(k,v)
    #         logger.debug(e)
    # logger.info('success')
    # os.remove('/home/amstel/llm/out/summarized_reviews.pkl')

