from loguru import logger
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections


if __name__ == '__main__':
    # Connect to Milvus
    connections.connect("default", host='localhost', port='19530')

    # Define the Milvus collection schema
    for text_entity in ("advantages", "disadvantages"):
        assert text_entity in ("advantages", "disadvantages")
        collection_name = f'summarize_product_reviews_embeddings_{text_entity}'
        fields = [
            FieldSchema(name="insertion_datetime", dtype=DataType.VARCHAR, max_length=26, description="insertion datetime"),
            FieldSchema(name="product_type", dtype=DataType.VARCHAR, max_length=50, description="e.g.: washing mashine"),
            FieldSchema(name="product_name", dtype=DataType.VARCHAR, is_primary=True, max_length=100, description="e.g.: LG 23W"),
            FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=25, description="name of embedding model"),
            FieldSchema(name=f"{text_entity}_embeddings", dtype=DataType.FLOAT_VECTOR, dim=768, description=f"embedded {text_entity}"),
        ]
        schema = CollectionSchema(fields, description="Summarized product reviews embeddings")
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(
            field_name=f"{text_entity}_embeddings",
            index_params={"index_type": "IVF_FLAT",
                            "params": {"nlist": 128},
                            "metric_type": "L2"}
        )
        collection.load()
    logger.info('Success')
