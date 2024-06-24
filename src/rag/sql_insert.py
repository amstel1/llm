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

if __name__ == '__main__':

    # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    dense_embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs={'device': 'cpu',})

    fields = [
        FieldSchema(name="q", dtype=DataType.VARCHAR, is_primary=True, max_length=1024,),
        FieldSchema(name="q_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="a", dtype=DataType.VARCHAR, max_length=1024),
    ]
    for sheet_name in ('washing_machine',  'fridge', 'tv', 'mobile', ):

        # milvus insert
        CONNECTION_URI = "http://localhost:19530"
        connections.connect(uri=CONNECTION_URI, db_name=sheet_name)


        print(sheet_name)
        collection = Collection(name='q_a_index', schema=CollectionSchema(fields=fields, enable_dynamic_field=False),)
        df = pd.read_excel('/home/amstel/llm/src/text2sql/examples.xlsx', sheet_name=sheet_name)
        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        collection.create_index("q_vector", dense_index)
        collection.flush()

        print('start embedding')
        entities = []
        for i, row in df.iterrows():
            rus_text = row['user']
            sql_text = row['sql']
            entity = {
                "q": rus_text,
                "q_vector": dense_embedding_model.embed_documents([rus_text])[0],
                "a": sql_text,
            }
            entities.append(entity)
        collection.insert(entities)
        collection.load()