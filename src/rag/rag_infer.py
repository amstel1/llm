import numpy as np
from loguru import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag_config import N_NEIGHBORS, SEPARATOR, TO_REPLACE_SEPARATOR, REPLACE_SEPARATOR_WITH, RAG_COLLECTION_NAME
from langchain_community.llms import Ollama

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def format_docs(docs):
    if TO_REPLACE_SEPARATOR:
        return "\n\n".join(doc.page_content.replace(SEPARATOR, REPLACE_SEPARATOR_WITH) for doc in docs)
    return "\n\n".join(doc.page_content for doc in docs)

def def_debugger(inp):
    logger.info(inp)
    return inp


if __name__ == '__main__':

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    db = Milvus(embedding_function=embedding_model, collection_name=RAG_COLLECTION_NAME,)
    retriever = db.as_retriever(search_kwargs={"k": N_NEIGHBORS, })

    llama_raw_template_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nТы - приветливый ИИ, разработанный в Сбер Банке (Беларусь). Ты знаешь только русский язык. Твоя задача - поддерживать диалог, отвечать на вопросы и тем самым помогать клиентам. Основываясь на контексте ниже, правдиво и полно отвечай на вопросы.<|eot_id|>"""
    llama_raw_template_user = """<|start_header_id|>user<|end_header_id|>\nКонтекст:\n\n{context}\n\nВопрос:\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    # Prompt
    debugger = RunnablePassthrough(def_debugger)
    prompt = PromptTemplate.from_template(template=llama_raw_template_system + llama_raw_template_user)


    # llm = LlamaCpp(
    #     model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf',
    #     n_gpu_layers=33,
    #     max_tokens=1024,
    #     n_batch=128,
    #     n_ctx=4096,
    #     f16_kv=True,
    #     verbose=False,
    #     temperature=0.0,
    #     stop=["<|eot_id|>", ],
    # )
    llm = Ollama(model="llama3_q6_32k", stop=["<|eot_id|>",], num_gpu=33, temperature=0, mirostat=0)
    # Chain
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | def_debugger
            | llm
            | StrOutputParser()
    )

    # Question
    # response = rag_chain.invoke("Какие есть кредиты для физических лиц?")
    # response = rag_chain.invoke("безотзывный депозит в белорусских рублях сохраняй, какие ставки?")
    # response = rag_chain.invoke("Какие есть карты для физических лиц?")
    # response = rag_chain.invoke("Сравни кредиты")
    # response = rag_chain.invoke("Сравни депозиты")
    # response = rag_chain.invoke("Сравни карты")
    #
    # response = rag_chain.invoke("Какой кредит самый выгодный?")
    # response = rag_chain.invoke("Какой депозит самый выгодный?")
    # response = rag_chain.invoke("Какая карта самая выгодная?")
    #
    # response = rag_chain.invoke("Подбери мне кредит")
    # response = rag_chain.invoke("Подбери мне депозит")
    # response = rag_chain.invoke("Подбери мне карту")
    # response = rag_chain.invoke("Подбери мне страховку")
    response = rag_chain.invoke("Собираюсь в отпуск в Турцию. Подбери мне страховку.")

    logger.info(f"response: {response}")