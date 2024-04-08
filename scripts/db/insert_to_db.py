
import pickle
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger
import itertools
CRAWL_ID = 2
if __name__ == '__main__':

    with open('../scripts/web_scraping/output.pkl', 'rb') as f:
        # tuple of 3 elements: product_page, breadcrumbs, item_list
        # we need item_list
        data = pickle.load(f)
    flattened_item_list = list(itertools.chain.from_iterable(data[-1]))
    df = pd.DataFrame(flattened_item_list)
    df['crawl_id'] = CRAWL_ID
    logger.warning(df.shape)
    user = 'scraperuser'
    password = 'scraperpassword'
    host = 'localhost'
    port = '6432'
    database = 'scraperdb'
    connection_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
    engine = create_engine(connection_str)
    try:
        with engine.connect() as connection_str:
            print('Successfully connected to the PostgreSQL database')
            df.to_sql(
                name='product_item_list',
                con=connection_str,
                schema='scraped_data',
                index=False,
                if_exists='append',
            )
    except Exception as ex:
        print(f'Sorry failed to connect: {ex}')