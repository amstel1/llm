import sys
sys.path.append('/home/amstel/llm/src')
from postgres.utils import PostgresDataFrameRead
from mongodb.utils import MongoRead
from typing import List, Dict, Iterable
from loguru import logger
# read entire tables



class DataServer():
    max_items = 4
    def __init__(self, product_type_name: str = 'washing_mashine'):
        self.product_type_name = product_type_name
        if product_type_name == 'washing_mashine':
            self.sql_details_db = 'scraped_data.washing_machine'
            self.sql_render_db = 'scraped_data.render_wm'  # render_washing_machine
            self.nosql_summarizations_db = 'scraped_data.product_review_summarizations'
        assert self.sql_details_db is not None
        assert self.sql_render_db is not None
        assert self.nosql_summarizations_db is not None

    def collect_one_item_data(self, name: str) -> Dict:
        results = {}

        postgres_reader = PostgresDataFrameRead(table=self.sql_details_db, where=f"name = '{name}'")
        sql_details = postgres_reader.read().get("step_0").sort_values('price', ascending=True).to_dict(orient='records')[0]
        if sql_details:
            results.update(sql_details)
        logger.debug(name)
        postgres_reader = PostgresDataFrameRead(table=self.sql_render_db, where=f"product_name = '{name}'")
        assert len(postgres_reader.read().get("step_0")) > 0
        sql_render = postgres_reader.read().get("step_0").to_dict(orient='records')[0]
        if sql_render:
            results.update(sql_render)

        mongo_reader = MongoRead(
            operation='read',
            db_name=self.nosql_summarizations_db.split('.')[0],
            collection_name=self.nosql_summarizations_db.split('.')[-1],
        )
        nosql_summarizations_dict = mongo_reader.read_one({"item_name":name})
        if nosql_summarizations_dict:
            results.update(nosql_summarizations_dict)
        return results

    def collect_data(self, names: Iterable) -> List[Dict]:
        results = []
        for i, name in enumerate(names):
            if i < self.max_items:
                item_data = self.collect_one_item_data(name)
                results.append(item_data)
            else:
                break
        return results