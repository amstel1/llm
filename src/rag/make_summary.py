# Забираем распарсенный документ (плохо структурирован), получаем саммари
import sys
from typing import Dict, Any

sys.path.append('/home/amstel/llm/src')
import numpy as np
from loguru import logger
import pandas as pd
import pickle
from langchain_core.documents import Document
from general_llm.langchain_llama_cpp_api_warpper import LlamaCppApiWrapper
from general_llm.llm_endpoint import call_generation_api, call_generate_from_history_api, call_generate_from_query_api, MODEL_NAME
from etl_jobs.base import Do, StepNum
from general_llm.prompt_construction import Llama3PromptTemplate, Gemma2PromptTemplate

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

class SberbankWebsiteSummaryDo(Do):
    def __init__(self, products: list[str] = ['deposits',]):
        self.products = products

    def process(self, data: Dict[StepNum, Any] = None) -> Dict[StepNum, list[Dict]]:
        assert data is None  # todo: rewrite so that both docs and df are read in the previous step
        all_results = {}
        for product in self.products:
            with open('/home/amstel/llm/src/web_scraping/bank_scraper/docs_all_02072024.pkl', 'rb') as f:
                docs = pickle.load(f)

            df = pd.read_excel(
                f'/home/amstel/llm/src/web_scraping/bank_scraper/{product}.xlsx')  # other_bank, credits, cards, dep
            agg = df.groupby('main')['aux'].apply(lambda x: list(x))

            df['aux'].value_counts().max() == 1

            aggregated_documents = []
            for source_link in agg.index:
                aux_links = agg.loc[source_link]
                contents = []
                for aux_link in aux_links:
                    contents.extend([x.page_content for x in docs if x.metadata.get('source') == aux_link])
                aggregated_documents.append(
                    Document(page_content='/n/n'.join(contents), metadata={'source': source_link}))

            with open(f'/home/amstel/llm/src/rag/aggregated_{product}_documents.pkl', 'wb') as f:
                pickle.dump(aggregated_documents, f)

            # with open('/home/amstel/llm/src/web_scraping/bank_scraper/docs_final.pkl', 'rb') as f:
            #     docs = pickle.load(f)
            with open(f'/home/amstel/llm/src/rag/aggregated_{product}_documents.pkl', 'rb') as f:
                docs = pickle.load(f)
            # to_scrape = process()
            # Initialize the LLM
            llm = LlamaCppApiWrapper()

            # todo: llama3 or gemma2
            if 'llama' in MODEL_NAME: prompt_template = Llama3PromptTemplate
            if 'gemma' in MODEL_NAME: prompt_template = Gemma2PromptTemplate

            if product == 'cards':
                system_prompt = "Ты эксперт в структурировании и суммаризации информации. Ты отлично распознаешь паттерны, очень внимателен к деталям, великолепен в выделении главного. В суждениях ты опираешься только на предоставленное Описание. "
                user_prompt_placeholder = "\nОписание:\n{input}\n\nВыше - описание банковской карты (или похожего банковского продукта). Извлеки из него ключевые условия / характеристики, например: точное название, валюта, срок действия, где и как открыть, условия money-back, стоимость оформления и использования. Результат должен быть кратким. Используй только русский язык. "

            elif product == 'credits':
                system_prompt = "Ты эксперт в структурировании и суммаризации информации. Ты отлично распознаешь паттерны, очень внимателен к деталям, великолепен в выделении главного. В суждениях ты опираешься только на предоставленное Описание. "
                user_prompt_placeholder = "\nОписание:\n{input}\n\nВыше - описание банковского кредитного продукта. Извлеки из него ключевые условия / характеристики, например: точное название, процентная ставка, сумма, срок, где и как открыть, прочие условия. Отдельно выдели информацию, если товар в кредит можно купить только у партнера. Результат должен быть кратким. Используй только русский язык. "

            elif product == 'other':
                system_prompt = "Ты эксперт в структурировании и суммаризации информации. Ты отлично распознаешь паттерны, очень внимателен к деталям, великолепен в выделении главного. В суждениях ты опираешься только на предоставленное Описание. "
                user_prompt_placeholder = "\nОписание:\n{input}\n\nВыше - описание банковского продукта или услуги. Извлеки из него все возомжные условия / характеристики. Результат должен быть кратким. Используй только русский язык."

            elif product == 'deposits':
                system_prompt = "Ты эксперт в структурировании информации. Ты отлично распознаешь паттерны, очень внимателен к деталям, великолепен в выделении улучшении структуры текста без искажений сути. В суждениях ты опираешься только на предоставленное Описание."
                user_prompt_placeholder = "\nОписание:\n{input}\n\nВыше - описание банковского депозита (вклада) или другого банковского продукта для накопления сбережений. Извлеки из него ключевые условия / характеристики, например: точное название, валюта, процентные ставки, срок, где и как открыть, отзывный / безотзывный. Если возможно, результат должен содержать все возможные комбинации валюты, срока, ставки. Результат должен быть кратким. Используй только русский язык. Удели максимальное внимание точности извлечения информации."



            results = {}
            for i, body in enumerate(docs):
                source_link = body.metadata.get('source')
                content = body.page_content
                assert len(content) > 10
                logger.debug(content)

                user_prompt = user_prompt_placeholder.format(input=content)
                formatting_prompt = prompt_template().create_prompt_from_user_query(
                   system_prompt_clean=system_prompt,
                   user_query=user_prompt
                )
                r = {}
                formatting_output = call_generation_api(prompt=formatting_prompt, )
                r['formatted'] = formatting_output
                # summarization_prompt = summarize_template.format(input=formatting_output)
                user_prompt_placeholder_summarization = '\nОписание:\n{input}\n\nИз описания выше извлеки название банковского продукта, о котором идет речь. Используй русский язык. Верни только само название продукта и ничего кроме. '
                user_prompt = user_prompt_placeholder_summarization.format(input=formatting_output)
                summarization_prompt = Llama3PromptTemplate().create_prompt_from_user_query(
                    system_prompt_clean='Ты наилучшим образом делаешь то что тебе говорят.',
                    user_query=user_prompt
                )
                summarization_output = call_generation_api(prompt=summarization_prompt,)
                r['summarized'] = summarization_output
                results[source_link] = r
                logger.info(f"{i}, {source_link}")
                logger.warning(r)
                print(r.get('formatted'))
                print()
                print(r.get('summarized'))
                print('=========')


            with open(f'rag_w_summary_results_{product}.pkl', 'wb') as f:
                pickle.dump(results, f)
            all_results[product] = results
        return {'step_0': all_results}


if __name__ == '__main__':
    summarizer = SberbankWebsiteSummaryDo()
    docs = summarizer.process()