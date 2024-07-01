# EMBEDDING
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
N_EMBEDDING_RESULTS = 20  # 20 for sberbank consultant, 5 for sql rag
ELBOW_EMBEDDING = True
# RERANKING
RERANKING_MODEL = 'BAAI/bge-reranker-v2-m3' # 'ms-marco-MultiBERT-L-12'  # ms-marco-MultiBERT-L-12 / rank-T5-flan
USE_RERANKER = True
N_RERANK_RESULTS = 10
ELBOW_RERANKING = True
RERANKING_THRESHOLD = 0.23

MOST_RELEVANT_AT_THE_TOP = True  # False means relevant result are close to the output, seems to work better

# RAG INDEXING
CHUNK_SIZE = 400                                # 500
CHUNK_OVERLAP = 50                             # 200
SPLITTER_SEPARATORS = ["\n", '**', '#', ]  # "\n\n\n"
RAG_COLLECTION_NAME = f'cards_{CHUNK_SIZE}_{CHUNK_OVERLAP}'

# OBSOLETE
SEPARATOR = ' || '                               # used in creating vectordb collection. will be used for managing context length in retriever
TO_REPLACE_SEPARATOR = False                    # False
REPLACE_SEPARATOR_WITH = ' '                    # ' '

