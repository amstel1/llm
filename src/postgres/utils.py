
import pickle
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger
import itertools
from datetime import datetime
from etl_jobs.base import Read, Write
from .config import user, password, host, port, database

class PostgresDataFrameWrite(Write):
    def __init__(self,

                schema_name: str = 'scraped_data',
                table_name: str = 'product_item_list',
                ):

        self.schema_name = schema_name
        self.table_name = table_name

    def write(
            self,
            data: pd.DataFrame
    ):
        logger.warning(data.shape)
        logger.info(data.head())

        connection_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
        engine = create_engine(connection_str)
        try:
            with engine.connect() as connection_str:
                print('Successfully connected to the PostgreSQL database')
                data.to_sql(
                    name=self.table_name,
                    con=connection_str,
                    schema=self.schema_name,
                    index=False,
                    if_exists='append',
                )
                logger.info('df.to_sql -- inserted successfully')
        except Exception as ex:
            logger.critical(f'Sorry failed to connect: {ex}')

class PostgresDataFrameRead(Read):
    def __init__(self,
                 table: str,
                 where: str = None):
        self.table = table
        self.where = where

    def read(
        self,
    ):
        #######################
        connection_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
        engine = create_engine(connection_str)
        #######################
        if self.where:
            sql_str = f'select * from {self.table} where {self.where};'
        else:
            sql_str = f'select * from {self.table};'
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