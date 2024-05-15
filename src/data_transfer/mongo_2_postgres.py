import sys
sys.path.append('/home/amstel/llm/src')
from mongodb.mongo_utils import MongoConnector
from postgres.utils import insert_data # todo: refactor
import pandas as pd
from datetime import datetime
from loguru import logger


if __name__ == '__main__':
    """Yandex market review-details (contains avg. rating & review cnt)"""
    # перекладывает from mongo to postgres

    # get from mongo
    mc = MongoConnector('read', 'scraped_data', 'product_details')
    response = list(mc.read_many({}))

    product_details = {}
    for review in response:
        review.pop('_id')
        remaining_key = list(review.keys())[0]
        review[remaining_key]['name'] = remaining_key  # user_query = key for joining
        product_details.update(review)

    product_details_df = pd.DataFrame(product_details).T
    product_details_df.to_excel('debug.xlsx')
    logger.debug(product_details_df['product_rating_value'].value_counts())
    product_details_df['product_rating_value'] = product_details_df['product_rating_value'].astype(float)
    product_details_df['product_rating_count'] = product_details_df['product_rating_count'].fillna(0).astype(int)
    product_details_df['product_review_count'] = product_details_df['product_review_count'].fillna(0).astype(int)
    product_details_df['inserted_datetime'] = datetime.now()

    # insert to postgres
    # todo: refactor
    insert_data(
        product_details_df,
        schema_name='scraped_data',
        table_name='reviews_product_details'
    )
    logger.info('success')