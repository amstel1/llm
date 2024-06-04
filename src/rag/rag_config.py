# RAG INFERCENCE & INDEXING
EMBEDDING_MODEL_NAME =  'BAAI/bge-m3'

# RAG INFERCENCE
N_EMBEDDING_RESULTS = 10  # embedding
N_RERANK_RESULTS = 5
USE_RERANKER = True
ELBOW = True  # elbow rule both for embedding and rerarnker
MOST_RELEVANT_AT_THE_TOP = False  # False means relevant result are close to the output
RERANKING_MODEL = 'BAAI/bge-reranker-v2-m3' # 'ms-marco-MultiBERT-L-12'  # ms-marco-MultiBERT-L-12 / rank-T5-flan


# RAG INDEXING
CHUNK_SIZE = 400                                # 500
CHUNK_OVERLAP = 0                             # 200
SPLITTER_SEPARATORS = ["\n", '**', '#', ]  # "\n\n\n"
RAG_COLLECTION_NAME = f'cards_{CHUNK_SIZE}_{CHUNK_OVERLAP}'

# OBSOLETE
SEPARATOR = ' '                               # ' | '
TO_REPLACE_SEPARATOR = False                    # False
REPLACE_SEPARATOR_WITH = ' '                    # ' '

