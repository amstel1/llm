# auth
# read
# write
# delete
# check_exists

import pymongo
import enum
from typing import Literal, List, Dict
MongoOperationType = Literal['write', 'read',]
MongoRole = Literal['reader', 'writer',]


CONFIG = {
    'host': 'localhost',
    'port': 27017,
    'credentials': {
        'reader': {'username': 'reader', 'password': 'reader'},
        'writer': {'username': 'writer', 'password': 'writer'},
    }
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
        self.client = pymongo.MongoClient(
            host=CONFIG.get('host'),
            port=CONFIG.get('port'),
            username=CONFIG.get('credentials').get(role).get('username'),
            password=CONFIG.get('credentials').get(role).get('password'),
        )
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def write_one(self, inserted_document: Dict):
        msg = self.collection.insert_one(inserted_document)
        return msg

    def write_many(self, inserted_documents: List[Dict]):
        msg = self.collection.insert_many(inserted_documents)
        return msg

    def read_one(self, key_value: Dict[str, str]):
        output_document = self.collection.find_one(key_value)
        return output_document

    def read_many(self, key_value: Dict[str, str]) -> pymongo.cursor.Cursor:
        cursor = self.collection.find(key_value)
        return cursor



