import pickle
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger
import itertools
from datetime import datetime

def insert_data(
        df:pd.DataFrame,
        schema_name: str = 'scraped_data',
        table_name: str = 'product_item_list',
):
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
                name=table_name,
                con=connection_str,
                schema=schema_name,
                index=False,
                if_exists='append',
            )
            logger.info('df.to_sql -- inserted successfully')
    except Exception as ex:
        logger.critical(f'Sorry failed to connect: {ex}')



def select_data(
    table: str,
    where: str = None
):
    #######################
    user = 'scraperuser'
    password = 'scraperpassword'
    host = 'localhost'
    port = '6432'
    database = 'scraperdb'
    connection_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
    engine = create_engine(connection_str)
    #######################

    if where:
        sql_str = f'select * from {table} where {where};'
    else:
        sql_str = f'select * from {table};'
    try:
        with engine.connect() as connection_str:
            logger.warning('Successfully connected to the PostgreSQL database')
            df = pd.read_sql(
                sql=sql_str,
                con=connection_str,
            )
            logger.warning(df.shape)
        return df
    except Exception as ex:
        logger.error(f'Sorry failed to connect: {ex}')
        return pd.DataFrame()


if __name__ == '__main__':
    raise AttributeError


    ### SELECT
    sql_from_table = 'scraped_data.product_item_list'
    where_clause = 'crawl_id <= 2'
    df = select_data(table=sql_from_table, where=where_clause)


    ########


    ### INSERT
    with open('/home/amstel/llm/out/shop_list.pkl', 'rb') as f:
        # tuple of 3 elements: product_page, breadcrumbs, item_list
        # we need item_list
        data = pickle.load(f)
    flattened_item_list = list(itertools.chain.from_iterable(data[-1]))
    df = pd.DataFrame(flattened_item_list)
    df['scraped_datetime'] = datetime.now()
    insert_data(df, schema_name='scraped_data', table_name='product_item_list')
    logger.warning('success')