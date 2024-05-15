SEPARATOR = ' '                               # ' | '
TO_REPLACE_SEPARATOR = False                    # False
REPLACE_SEPARATOR_WITH = ' '                    # ' '
N_NEIGHBORS = 8                       # 1 - 3 - 5 - 6

CHUNK_SIZE = 400                                # 500
CHUNK_OVERLAP = 0                             # 200
EMBEDDING_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v1"


RAG_COLLECTION_NAME = f'other_{CHUNK_SIZE}_{CHUNK_OVERLAP}'
# RAG_COLLECTION_NAME = f'recursive_collection_500_250'
# RAG_COLLECTION_NAME = f'recursive_collection_800_300'
# RAG_COLLECTION_NAME = f'md_collection'

SPLITTER_SEPARATORS = ["\n", '**', '#', ]  # "\n\n\n"