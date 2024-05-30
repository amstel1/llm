from langchain_community.retrievers import MilvusRetriever
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import Milvus

if __name__ == '__main__':
    embedding_model = LlamaCppEmbeddings(
        model_path='/home/amstel/llm/models/Publisher/Repository/nomic-embed-text-v1.5.Q8_0.gguf',
        n_gpu_layers=13,
        n_batch=64,
        verbose=False,
    )
    db = Milvus(
        embedding_function=embedding_model,
        collection_name="summarize_product_reviews_embeddings_advantages",
    )
    retriever = db.as_retriever()
    
    # to view the distances
    # db.similarity_search_with_score("громкий набор воды", k=2)

    # docs: List[Document], Document.page_content:str, Document.metadata: Dict[str,str]
    docs = retriever.invoke("Хорошая стиральная машина")
