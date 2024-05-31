# todo: decide between reranking models

# RAG INFERCENCE & INDEXING
EMBEDDING_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v1"

# RAG INFERCENCE
N_NEIGHBORS = 6
N_RERANK_RESULTS = 6
USE_RERANKER = False
RERANKING_MODEL = 'ms-marco-MultiBERT-L-12'  # ms-marco-MultiBERT-L-12 / rank-T5-flan

# RAG INDEXING
CHUNK_SIZE = 400                                # 500
CHUNK_OVERLAP = 0                             # 200
SPLITTER_SEPARATORS = ["\n", '**', '#', ]  # "\n\n\n"
RAG_COLLECTION_NAME = f'cards_{CHUNK_SIZE}_{CHUNK_OVERLAP}'

# OBSOLETE
SEPARATOR = ' '                               # ' | '
TO_REPLACE_SEPARATOR = False                    # False
REPLACE_SEPARATOR_WITH = ' '                    # ' '

