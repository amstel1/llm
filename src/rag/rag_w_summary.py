# todo: implemetnt vector search
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag_config import SPLITTER_SEPARATORS, CHUNK_OVERLAP, CHUNK_SIZE, RAG_COLLECTION_NAME
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.retrievers import ParentDocumentRetriever
import numpy as np
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence
import pandas as pd
import pickle


# def text_cosine_similarity(text_a: str, text_b: str):
#     a = embedding_model.embed_query(text_a)
#     b = embedding_model.embed_query(text_b)
#     return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def process():
    with open('/home/amstel/llm/src/web_scraping/bank_scraper/tree.pkl', 'rb') as f:
        tree = pickle.load(f)
    concat_vals = np.concatenate([x for x in tree.values() if x]).tolist()
    concat_vals_b = list(tree.keys())
    concat_vals += concat_vals_b
    final = []
    for link in concat_vals:
        if link.startswith('https://www.sber-bank.by') and \
                (not link.endswith('.pdf')) and \
                (not link.endswith('.docx')) and \
                (not link.endswith('.doc')) and \
                (not link.endswith('.xlsx')) and \
                (not link.endswith('.apk')) and \
                (not link.endswith('.zip')) and \
                (not link.endswith('.rar')) and \
                (not link.endswith('.svg')) and \
                (not link.endswith('.xls')):
            final.append(link)

    final = [x for x in final if 'news' not in x]
    final = [x for x in final if 'business' not in x]
    final = [x for x in final if 'biznes' not in x]
    final = [x for x in final if '+375' not in x]
    final = [x for x in final if '#_ftn' not in x]
    final = [x for x in final if 'files/up' not in x]
    final = [x for x in final if 'loginsbol' not in x]
    final = [x for x in final if '?selected-insurance-id=' not in x]
    final = [x for x in final if 'token-scope' not in x]
    final = [x for x in final if '?selected-credit' not in x]
    final = [x for x in final if '?request=' not in x]
    final = [x for x in final if 'Dostavych' not in x]
    final = [x for x in final if 'zarplatnyj-proekt' not in x]
    final = [x for x in final if 'rabota-s-nalichnostyu' not in x]
    final = [x for x in final if 'mezhdunarodnye' not in x]
    final = [x for x in final if 'packet' not in x]
    final = [x for x in final if 'currency_exchange_trade' not in x]
    final = [x for x in final if 'property_sale' not in x]
    final = [x for x in final if 'page' not in x]
    vc = pd.Series(final).value_counts()
    vc = vc[vc >= 6]
    to_scrape = vc.index.tolist()
    to_scrape.extend(['https://www.sber-bank.by/card/ultracard-2-0', 'https://www.sber-bank.by/card/BELKART-PREMIUM'])
    return to_scrape


if __name__ == '__main__':

    with open('/home/amstel/llm/src/web_scraping/bank_scraper/docs_final.pkl', 'rb') as f:
        docs = pickle.load(f)
    to_scrape = process()

    # Initialize the LLM
    llm = Ollama(model="llama3", stop=["<|eot_id|>"], num_gpu=-1, num_thread=-1, temperature=0, mirostat=0)

    # Define the templates
    llama_raw_template_system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Ты наилучшим образом делаешь то что тебе говорят.<|eot_id|>"
    llama_raw_template_user = "<|start_header_id|>user<|end_header_id|>\nКонтекст:\n\n{context}\n\nВопрос:\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    # Create the formatting prompt
    formatting_prompt = PromptTemplate.from_template(template=llama_raw_template_system + llama_raw_template_user)

    # Define the formatting chain
    formatting_chain = formatting_prompt | llm | StrOutputParser()

    # Define the summarization template and prompt
    summarize_template = "<|start_header_id|>user<|end_header_id|>Описание:\n{input}\n\nВыше приведено описание банковского продукта или услуги. Напиши одно предложение которое максимально точно и полно характериует его суть и преимущества. Отвечай на русском языке.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    summarization_prompt = PromptTemplate.from_template(template=summarize_template)

    # Define summarization chain, ensuring input is formatted as expected
    summarization_chain = {"input": lambda x: {"input": x}} | summarization_prompt | llm | StrOutputParser()

    combined_chain = RunnableParallel({
        "formatted":formatting_chain,
        "summarized":summarization_chain
    })

    results = {}
    for i, source_link in enumerate(to_scrape):
        filtered = [doc for doc in docs if doc.metadata.get('source') == source_link]
        content = filtered[0].page_content
        assert len(content) > 10
        r = combined_chain.invoke({
            "question": "Выше приведено содержимое вебсайта. Данные извлечены из каждого тега и разделены символом | . Сделай эти данные более структурированными и соедини релевантные параграфы вместе. Ты можешь только менять форматирование документа. Отвечай на русском языке.",
            "context": content,
        })
        results[source_link] = r
        logger.info(f"{i}, {source_link}")

    with open('rag_w_summary_results.pkl', 'wb') as f:
        pickle.dump(results, f)