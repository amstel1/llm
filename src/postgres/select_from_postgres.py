import pandas as pd
from sqlalchemy import create_engine
from loguru import logger

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
    sql_from_table = 'scraped_data.product_item_list'
    where_clause = 'crawl_id <= 2'
    df = select_data(table=sql_from_table, where=where_clause)
