# todo: implemetnt vector search
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings

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
from langchain_core.documents import Document

# def text_cosine_similarity(text_a: str, text_b: str):
#     a = embedding_model.embed_query(text_a)
#     b = embedding_model.embed_query(text_b)
#     return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def process(slug='credit'):
    # slug is a word that is present in
    # '.by/credit', '.by/card', '.by/deposit', '.by/insurance', '.by/vklady', '.by/investicii', '.by/sberbank_first', '.by/strah', '.by/page'
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
    vc = pd.Series(final).value_counts()
    vc = vc.loc[[x for x in vc.index.tolist() if slug in x]]
    to_scrape = vc.index.tolist()
    to_scrape.extend(['https://www.sber-bank.by/card/ultracard-2-0', 'https://www.sber-bank.by/card/BELKART-PREMIUM'])
    return to_scrape


if __name__ == '__main__':
    with open('/home/amstel/llm/src/web_scraping/bank_scraper/docs_deposits_10052024.pkl', 'rb') as f:
        docs = pickle.load(f)

    df = pd.read_excel('/home/amstel/llm/src/web_scraping/bank_scraper/cre.xlsx')

    agg = df.groupby('main')['aux'].apply(lambda x: list(x))

    df['aux'].value_counts().max() == 1

    aggregated_documents = []
    for source_link in agg.index:
        aux_links = agg.loc[source_link]
        contents = []
        for aux_link in aux_links:
            contents.extend([x.page_content for x in docs if x.metadata.get('source') == aux_link])
        aggregated_documents.append(Document(page_content='/n/n'.join(contents), metadata={'source': source_link}))

    with open('aggregated_deposit_documents.pkl', 'wb') as f:
        pickle.dump(aggregated_documents, f)

    # with open('/home/amstel/llm/src/web_scraping/bank_scraper/docs_final.pkl', 'rb') as f:
    #     docs = pickle.load(f)
    with open('/home/amstel/llm/src/rag/aggregated_deposit_documents.pkl', 'rb') as f:
        docs = pickle.load(f)
    # to_scrape = process()
    # Initialize the LLM
    llm = Ollama(model="llama3_q6_32k", stop=["<|eot_id|>"], num_gpu=-1, num_thread=-1, temperature=0, mirostat=0)


    # Define the summarization template and prompt
    formatting_template = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                          # "You are tasked with compessing the content from the user. This informtaion will later be used to judge if the content is worthy and viable. You must make it readable and coherent. Save the full extent of the details. You can omit useless information.<|eot_id|>"
                          " Ты знаешь только русский язык. Ты эксперт в структурировании и суммаризации информации. Ты отлично распознаешь паттерны, крайне внимателен к деталям и великолепен в выделении главного. В суждениях ты всегда опираешься на предоставленную информацию.<|eot_id|>"             
                          "<|start_header_id|>user<|end_header_id|>\n"
                          # "below is a description of a bank product."
                          # " extract key features including but not limited to: full name, currency, percentage rate, term, where to open, if it can be recalled."
                          # " if possible the resut must contain all possible combinations of currency, term, rate. используй только русский язык."
                          " Ниже - описание банковского продукта."
                          " Извлеки из него ключевые характеристики включая, но не ограничиваясь: название, валюта, ставка процента, срок, где открыть, отзывный / безотзывный."
                          " Если возможно результат должен содержать все возможные комбинации валюты, срока, ставки. Результат должен быть не длинее 1000 символов. Используй только русский язык."
                          "\n\n описание:\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
    formatting_prompt = PromptTemplate.from_template(template=formatting_template)

    # Define summarization chain, ensuring input is formatted as expected
    formatting_chain = formatting_prompt | llm | StrOutputParser()
    summarize_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Описание:\n{input}\n\nИз описания выше извлеки краткое название банковского продукта, о котором идет речь. Используй исключительно русский язык. Верни только само название продукта и ничего кроме.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    summarization_prompt = PromptTemplate.from_template(template=summarize_template)
    summarization_chain = {"input": lambda x: {"input": x}} | summarization_prompt | llm | StrOutputParser()

    combined_chain = RunnableParallel({
        "formatted":formatting_chain,
        "summarized":summarization_chain
    })

    results = {}
    for i, body in enumerate(docs):
        source_link = body.metadata.get('source')
        content = body.page_content
        # filtered = [doc for doc in docs if doc.metadata.get('source') == source_link]
        # content = filtered[0].page_content
        assert len(content) > 10
        # logger.info(content)
        r = combined_chain.invoke({
            # "question": "Сделай данные выше более структурированными, соедини релевантные параграфы вместе. Результат должен быть кратким, содержать название и ключевые параметры / условия продукта, например: валюта, ставка, срок, сумма, процент. Ты можешь только менять форматирование документа. Используй русский язык.",
            "input": content,
        })
        results[source_link] = r
        logger.info(f"{i}, {source_link}")
        logger.warning(r)
        # print(r.get('formatted'))
        # print()
        # print(r.get('summarized'))
        print('=========')


    with open('rag_w_summary_results_deposits2.pkl', 'wb') as f:
        pickle.dump(results, f)