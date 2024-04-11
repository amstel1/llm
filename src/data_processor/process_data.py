import sys
import os
sys.path.append('/home/amstel/llm/src')
from typing import Dict, Tuple, List
from mongodb.utils import MongoConnector
import pickle
from loguru import logger

class YandexMarketProcessor:
    """product details and reviews"""
    @logger.catch
    @staticmethod
    def process(data: Dict) -> Tuple[List[Dict[str, Dict]], List[Dict[str, List[Dict]]]]:
        '''
        returns tuple(products, reviews)
        insert the result to mongo
        '''
        product_details_output = []
        reviews_details_output = []
        for product_name, combined_data in data.items():
            assert product_name == combined_data[0]
            product_tuple = combined_data[1]
            reviews_tuple = combined_data[2]
            if product_tuple:
                product_request_url, product_details = product_tuple
                if product_details:
                    product_details['product_request_url'] = product_request_url
                    product_details_output.append({product_name: product_details})  # str -> Dict

            if reviews_tuple:
                reviews_request_url, reviews_details = reviews_tuple
                if reviews_details:
                    # reviews_details is list
                    for review in reviews_details:
                        review.update({'reviews_request_url': reviews_request_url})
                    reviews_details_output.append({product_name: reviews_details})  # str -> List[Dict]

        return product_details_output, reviews_details_output


if __name__ == '__main__':
    # parsed yandex market, insert to pymongo
    with open('/home/amstel/llm/out/future_pairs.pkl', 'rb') as f:
        pairs = pickle.load(f)
    product_details, reviews_details = YandexMarketProcessor.process(data=pairs)

    cursor = MongoConnector(operation='write', db_name='scraped_data', collection_name='product_details')
    cursor.write_many(product_details)

    logger.info('success')