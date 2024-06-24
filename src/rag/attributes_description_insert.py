from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
)
import numpy as np
import sys
sys.path.append('/home/amstel/llm/src')
from postgres.utils import PostgresDataFrameRead
from etl_jobs import attribute_mappings


class DescriptionMaker:
    def __init__(self, schema_name: str):
        assert schema_name in ('washing_machine', 'fridge', 'mobile', 'tv', )
        name_2_attribute = {
            'washing_machine': attribute_mappings.washing_machine_mapping,
            'fridge': attribute_mappings.fridge_mapping,
            'mobile': attribute_mappings.mobile_mapping,
            'tv': attribute_mappings.tv_mapping
        }
        self.attribute_mapping = name_2_attribute[schema_name]  # key (rus), value (eng)
        self.attribute_mapping.update({"цена":"min_price", "name":"название товара"})
        self.postgres_reader = PostgresDataFrameRead(table=f'{schema_name}.{schema_name}')

    def make(self):
        #  "brand" text, -- название производителя
        # return {attr_name(eng): (type, rus)}
        result = {}
        df = self.postgres_reader.read().get('step_0')
        print(df.head(1))
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        string_cols = [x for x in df.columns if x in df.select_dtypes(include=[object]).columns.tolist() and x not in numeric_cols]
        inverse_attribute_mapping = {v: k for k,v in self.attribute_mapping.items()}  # eng -> rus
        for eng_col, rus_col in inverse_attribute_mapping.items():
            if eng_col in numeric_cols:
                this_dtype = 'real'
            else:
                this_dtype = 'text'
            result[eng_col] = (this_dtype, rus_col)
        return result


if __name__ == '__main__':

    # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    dense_embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs={'device': 'cpu',})

    fields = [
        FieldSchema(name="attribute_name_eng", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="attribute_name_rus", dtype=DataType.VARCHAR, is_primary=True, max_length=1024,),
        FieldSchema(name="attribute_name_rus_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="attribute_type", dtype=DataType.VARCHAR, max_length=4),  # may only be real / text1
    ]
    for db_name in ( 'fridge', 'tv', 'mobile', 'washing_machine', ):

        # milvus insert
        CONNECTION_URI = "http://localhost:19530"
        connections.connect(uri=CONNECTION_URI, db_name=db_name)


        print(db_name)
        collection = Collection(name='postgres_table_attributes', schema=CollectionSchema(fields=fields, enable_dynamic_field=False),)
        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        collection.create_index("attribute_name_rus_vector", dense_index)
        collection.flush()


        eng_2_rus = DescriptionMaker(schema_name=db_name).make()
        print('start embedding')
        entities = []
        for key, vals in eng_2_rus.items():
            eng = key
            dtype, rus = vals
            entity = {
                "attribute_name_eng": eng,
                "attribute_name_rus": rus,
                "attribute_name_rus_vector": dense_embedding_model.embed_documents([rus])[0],
                "attribute_type": dtype,
            }
            entities.append(entity)
        collection.insert(entities)
        collection.load()