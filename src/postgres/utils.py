import sys
sys.path.append('/home/amstel/llm/src')
import pickle
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger
import itertools
from datetime import datetime
from etl_jobs.base import Read, Write
from postgres.config import user, password, host, port, database
from typing import List, Dict, Tuple, Any
from etl_jobs.base import StepNum

class PostgresDataFrameWrite(Write):
    def __init__(self,
                schema_name: str = 'scraped_data',
                table_name: str = 'product_item_list',
                insert_unique: bool = True,  # whether to insert unique
                index_column: str = "product_url",  # column to check uniqueness against
                ):

        self.schema_name = schema_name
        self.table_name = table_name
        self.insert_unique = insert_unique
        self.index_column = index_column

    def write(
            self,
            data: Dict[StepNum, pd.DataFrame],
    ):
        assert isinstance(data, dict)
        data = data.get("step_0")
        logger.warning(data.shape)
        logger.info(data.head())

        connection_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
        engine = create_engine(connection_str)
        try:
            with engine.connect() as connection_str:
                print('Successfully connected to the PostgreSQL database')
                if self.insert_unique:
                    try:
                        df = pd.read_sql(
                            sql=f'select * from {self.schema_name}.{self.table_name}',
                            con=connection_str,
                        )
                        if self.index_column in df.columns and self.index_column in data.columns:
                            data = data[~data[self.index_column].isin(df[self.index_column])]  # only unique remain
                            logger.warning(f'after unique -- {data.shape}')
                    except Exception as e:
                        logger.error(f"error -- insertion uniqueness not satisfied: {e}")
                try:
                    logger.warning(data.dtypes)
                    data.to_sql(
                        name=self.table_name,
                        con=connection_str,
                        schema=self.schema_name,
                        index=False,
                        if_exists='append',
                    )
                    logger.info(f'df.to_sql -- inserted successfully -- {data.shape} to {self.schema_name}.{self.table_name}')
                except Exception as e:
                    logger.error(f"error -- insert sql failed: {e}")
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
    ) -> Dict[StepNum, Any]:
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
            return {"step_0": df}
        except Exception as ex:
            logger.error(f'Sorry failed to connect: {ex}')
            return {"step_0": pd.DataFrame()}


if __name__ == '__main__':
    data = pd.read_pickle('/home/amstel/llm/src/etl_jobs/datastep_0.pkl')
    print(data.head(1).T)
    print('#####')
    print(data.head(1)['product_type_url'].values)
    print('#####')
    print(data.shape)
    print(data.dtypes)
    w = PostgresDataFrameWrite(
        schema_name='scraped_data',
        table_name='product_item_list_to_fill',
        insert_unique=True,
        index_column="product_url",
    )
    w.write({"step_0":data})