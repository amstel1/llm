SEPARATOR = ' | '                               # ' | '
TO_REPLACE_SEPARATOR = False                    # False
REPLACE_SEPARATOR_WITH = ' '                    # ' '
N_NEIGHBORS = 5                           # 1 - 3 - 5 - 6

CHUNK_SIZE = 750                                # 500
CHUNK_OVERLAP = 100                             # 200

RAG_COLLECTION_NAME = f'markdown_MiniLM_{CHUNK_SIZE}_{CHUNK_OVERLAP}'                # semantic embedding, default prameters
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # / "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


SPLITTER_SEPARATORS = [ "\n\n", "\n", '|', ]  # "\n\n\n"