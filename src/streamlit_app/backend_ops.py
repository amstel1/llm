import sys
sys.path.append('/home/amstel/llm/src')
from postgres.utils import PostgresDataFrameRead
from mongodb.utils import MongoRead
from typing import List, Dict, Iterable
from loguru import logger
# read entire tables



class DataServer():
    max_items = 4
    def __init__(self, schema_name: str):
        product_type_name = schema_name
        assert product_type_name in ('washing_machine', 'fridge', 'tv', 'mobile')
        self.product_type_name = product_type_name
        # logger.critical(product_type_name)
        self.sql_details_db = f'{product_type_name}.{product_type_name}'
        self.sql_render_db = f'{product_type_name}.render_{product_type_name}'  # render_washing_machine
        self.nosql_summarizations_db = f'{product_type_name}.product_review_summarizations'
        assert self.sql_details_db is not None
        assert self.sql_render_db is not None
        assert self.nosql_summarizations_db is not None

    def collect_one_item_data(self, name: str) -> Dict:
        results = {}
        # logger.warning(f'!!!  sql_details  1 -  {self.sql_details_db}, {name}')
        postgres_reader = PostgresDataFrameRead(table=self.sql_details_db, where=f"name = '{name}'")
        sql_details = postgres_reader.read().get("step_0").sort_values('price', ascending=True).to_dict(orient='records')[0]
        # logger.warning(f'!!!  sql_details 2 - {sql_details}')
        if sql_details:
            results.update(sql_details)
        # logger.warning(f'!!!  results 1 - {results}')
        ###############
        # logger.debug(name)
        ###############
        postgres_reader = PostgresDataFrameRead(table=self.sql_render_db, where=f"name = '{name}'")
        try:
            postgres_reader_results = postgres_reader.read().get("step_0")
            assert len(postgres_reader_results) > 0
            sql_render = postgres_reader_results.to_dict(orient='records')[0]
        except AssertionError:
            sql_render = {}

        if sql_render:
            results.update(sql_render)
        # logger.warning(f'!!!  results 2 - {results}')
        mongo_reader = MongoRead(
            operation='read',
            db_name=self.nosql_summarizations_db.split('.')[0],
            collection_name=self.nosql_summarizations_db.split('.')[-1],
        )
        nosql_summarizations_dict = mongo_reader.read_one({"item_name": name})
        if nosql_summarizations_dict:
            results.update(nosql_summarizations_dict)
        # logger.warning(f'!!!  results 3 - {results}')
        return results

    def collect_data(self, names: Iterable) -> List[Dict]:
        results = []
        for i, name in enumerate(names):
            if i < self.max_items:
                item_data = self.collect_one_item_data(name)
                # logger.warning(f'one item data: {item_data}')
                if item_data:
                    results.append(item_data)
            else:
                break
        return results


if __name__ == '__main__':
    name = 'Мобильный телефон BQ BQ-2446 Dream Duo'
    schema_name = 'mobile'
    ds = DataServer(schema_name=schema_name)
    item = ds.collect_one_item_data(name=name)
    print(item)