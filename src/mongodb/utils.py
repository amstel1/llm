import sys
sys.path.append('/home/amstel/llm/src')
from etl_jobs.base import Read, Write, StepNum
import pymongo
from typing import Literal, List, Dict, Any
from loguru import logger
import numpy as np
import pickle
MongoOperationType = Literal['write', 'read',]
MongoRole = Literal['reader', 'writer',]


CONFIG = {
    'host': 'localhost',
    'port': 27017,
}

CREDENTIALS = {
    'reader': {'username': 'reader', 'password': 'reader'},
    'writer': {'username': 'writer', 'password': 'writer'},
}
class MongoConnector:
    def __init__(self, operation: MongoOperationType, db_name: str, collection_name: str):
        self.operation = operation
        self.db_name = db_name
        self.collection_name = collection_name
        if operation in ('read', 'check_exists'):
            self.auth(role='reader')
        elif operation in ('write', 'delete'):
            self.auth(role='writer')
        assert self.db is not None
        assert self.collection is not None

    def auth(self, role: MongoRole):
        username = CREDENTIALS.get(role).get('username')
        password = CREDENTIALS.get(role).get('password')
        self.client = pymongo.MongoClient(
            f"mongodb://{username}:{password}@{CONFIG.get('host')}:{CONFIG.get('port')}/"
        )
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def write_one(self, inserted_document: Dict):
        try:
            msg = self.collection.insert_one(inserted_document)
            return msg
        except Exception as e:
            logger.critical(e)

    def write_many(self, inserted_documents: List[Dict]):
        try:
            msg = self.collection.insert_many(inserted_documents, ordered=False)
            return msg
        except Exception as e:
            logger.critical(e)

    def read_one(self, key_value: Dict[str, str]):
        output_document = self.collection.find_one(key_value)
        return output_document

    def read_many(self, key_value: Dict[str, str]) -> pymongo.cursor.Cursor:
        cursor = self.collection.find(key_value)
        return cursor

class MongoRead(Read):
    def __init__(self, operation: MongoOperationType, db_name: str, collection_name: str):
        self.operation = operation
        self.db_name = db_name
        self.collection_name = collection_name
        self.mongo_connector = MongoConnector(operation=self.operation, db_name=self.db_name,
                                         collection_name=self.collection_name)
        # if operation in ('read', 'check_exists'):
        #     self.mongo_connector.auth(role='reader')
        # elif operation in ('write', 'delete'):
        #     self.mongo_connector.auth(role='writer')
        assert self.mongo_connector.db is not None
        assert self.mongo_connector.collection is not None

    def read(self) -> Dict[StepNum, List[str]]:
        cursor = self.mongo_connector.read_many({})
        return {"step_0": list(cursor)}

    def read_one(self,  key_value: Dict[str, str]):
        output_document = self.mongo_connector.read_one(key_value)
        return output_document

class MongoWrite(Write):
    def __init__(self, operation: MongoOperationType, db_name: str, collection_name: str):
        self.operation = operation
        self.db_name = db_name
        self.collection_name = collection_name
        self.mongo_connector = MongoConnector(operation=self.operation, db_name=self.db_name,
                                              collection_name=self.collection_name)
        if operation in ('read', 'check_exists'):
            self.mongo_connector.auth(role='reader')
        elif operation in ('write', 'delete'):
            self.mongo_connector.auth(role='writer')
        assert self.mongo_connector.db is not None
        assert self.mongo_connector.collection is not None

    def write(self, data: Any) -> None:
        if len(data) > 0:
            if all(isinstance(x,list) for x in data):
                logger.debug("debug mongo - ravel")
                logger.debug(data)
                try:
                    data = list(np.ravel(data))
                except Exception as e:
                    data = [item for sublist in data for item in sublist]
        if len(data) > 0:
            # self.mongo_connector.write_many(data)
            for i, element in enumerate(data):
                try:
                    self.mongo_connector.write_one(element)
                    logger.debug(i)
                except Exception as e:
                    logger.debug(e)
            # logger.info(f'written: {data}')
        # cursor = MongoConnector(operation='write', db_name='scraped_data', collection_name='product_details')
        # cursor.write_many(product_details)
        #
        # cursor = MongoConnector(operation='write', db_name='scraped_data', collection_name='product_reviews')
        # cursor.write_many(reviews_details)

def delete_empty():
    role = 'write'
    query = {"$and": [{"_id": {"$exists": True}},
                      {"$expr": {"$eq": [{"$objectToArray": "$$ROOT"}, [{"k": "_id", "v": "$_id"}]]}}]}
    CONFIG = {
        'host': 'localhost',
        'port': 27017,
    }
    CREDENTIALS = {
        'reader': {'username': 'reader', 'password': 'reader'},
        'writer': {'username': 'writer', 'password': 'writer'},
    }
    username = CREDENTIALS.get(role).get('username')
    password = CREDENTIALS.get(role).get('password')
    client = pymongo.MongoClient(
        f"mongodb://{username}:{password}@{CONFIG.get('host')}:{CONFIG.get('port')}/"
    )
    db = client['fridge']
    collection = db['product_details']
    result = collection.delete_many(query)
    print(result)


if __name__ == '__main__':

    # READ 1
    # con_product_reviews = MongoConnector(operation='read', db_name='scraped_data', collection_name='product_reviews')
    # cursor_product_reviews = con_product_reviews.read_many({})
    # product_reviews = list(cursor_product_reviews)
    #
    # # READ 2
    # con_product_details = MongoConnector(operation='read', db_name='scraped_data', collection_name='product_details')
    # cursor_product_details = con_product_details.read_many({})
    # product_details = list(cursor_product_details)

    with open('/home/amstel/llm/out/summarized_reviews.pkl', 'rb') as f:
        summarized_reviews = pickle.load(f)
    con_product_details = MongoConnector(
        operation='write',
        db_name='scraped_data',
        collection_name='product_review_summarizations'
    )
    cursor_product_details = con_product_details.write_many(summarized_reviews)
    # product_details = list(cursor_product_details)