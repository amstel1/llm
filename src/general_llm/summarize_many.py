# todo: dynamically gemerate FRAGMENT_LENGHT_LIMIT depending on the length of advantages / disadvantages
FRAGMENT_LENGHT_LIMIT = 1500
import sys
sys.path.append('/home/amstel/llm')
import pickle
from src.mongodb.utils import MongoConnector
from loguru import logger
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


import os
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from datetime import datetime
trace_name = f'summary_llama3_{datetime.now()}'
print(datetime.now())
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-a45f2a3d-4085-4170-b337-8cc2f1921aef"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-26acbd19-74af-4cb4-93e1-919526c13921"
langfuse = Langfuse()
langfuse_callback_handler = CallbackHandler(trace_name=trace_name)

import json
from typing import List
import sys
sys.path.append('/home/amstel/llm/src')
from mongodb.utils import MongoConnector
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import AIMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# second best -- looks like this may be the better prompt
# mistral_raw_template_system = '<s> [INST] \n Ты эксперт мирового уровня по анализу отзывов на товары. Выдели главные преимущества и недостатки товара из отзывов ниже. Отвечай только на русском языке. Ты всегда должен сформировать ответ в виде JSON с ключами "advantages" и "disadvantages".\n [/INST]' \

# best - idk
# mistral_raw_template_system = '<s> [INST] Ты эксперт мирового уровня по анализу отзывов на товары. В качестве ответа ты должен вернуть только JSON с ровно двумя ключами: "достоинства", "недостатки". Значения должны содержать список достоинств и недостатков данного товара, которые указаны в отзывах. [/INST] </s> \n\n'
# mistral_raw_template_context = ("<s> [INST] Отзывы: {context}. [/INST] </s> \n\n")

# mistral_raw_template_context = "<s> [INST] \n Отзывы: {context}.\n [/INST]" # v3

llama_raw_template_system_ru = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты эксперт мирового уровня по анализу отзывов на товары. Cуммаризуй текст ниже. Отвечай ТОЛЬКО на русском языке. Ты всегда должен формировать ответ в виде JSON с одним из ключей: "advantages", "disadvantages".<|eot_id|>"""

llama_raw_template_system_a_en = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Summarize the text below in a truthful, eloquent manner. Return a valid JSON blob with key "advantages" and corresponding values as a list of strings in Russian. Do not use enumerated lists. No mentioning "yandex" or "market".<|eot_id|>"""

llama_raw_template_system_d_en = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Summarize the text below in a truthful, eloquent manner. Return a valid JSON blob with key the "disadvantages" corresponding to the summarized disadvantages as a list of strings in Russian only. If there are none, return empty list. No enumerations. No mentioning "yandex" or "market".<|eot_id|>"""

# llama_raw_template_system_c_en = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# Summarize the text below in a truthful, eloquent manner. Return only a valid JSON blob with key "comments" and corresponding values as a list of strings in Russian. If there are none, return empty list. Do not use enumerations.<|eot_id|>"""

llama_raw_template_user = """<|start_header_id|>user<|end_header_id|>\nОтзывы:\n\n{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>JSON: """

def strip_latin(input: str) -> str:
    latin_lower = 'qwertyuiopasdfghjklzxcvbnm'
    latin_upper = latin_lower.upper()
    latin_chars = latin_lower + latin_upper
    trans_table = str.maketrans({x:'' for x in list(latin_chars)})
    return input.translate(trans_table)

def pase_review(review: str) -> tuple[str, str, str]:
    '''returns: (advantages, disadvantages, comments)'''
    advantages_pos = review.find('Достоинства:')
    disadvantages_pos = review.find('Недостатки:')
    comments_pos = review.find('Комментарий:')
    return (
        review[advantages_pos:disadvantages_pos],
        review[disadvantages_pos:comments_pos],
        review[comments_pos:],
    )


def unite_same_type(fragments: list[str], fragment_length_limit: int = 1500, separator: str='\n|\n') -> list[str]:
    """возвращает список объединенных фрагментов, каждый из которых не больше fragment_length_limit"""
    output_list = []
    n_total_fragments = len(fragments)
    for i, fragment in enumerate(fragments):
        if i == 0:
            s = fragment
            len_so_far = len(s)
        else:
            if len_so_far + len(fragment) <= fragment_length_limit:
                s += separator
                s += fragment
                len_so_far = len(s)
            else:
                output_list.append(s)
                s = fragment
                len_so_far = len(s)
        if i == n_total_fragments - 1:
            output_list.append(s)
    return output_list


def get_united_types(reviews: list[str]) -> tuple[list[str], list[str], list[str]]:
    '''reviews is list[str] for ONE product'''
    advantages, disadvantages, commtents = [], [], []
    for review in reviews:
        advnt, dsadvnt, cmnt = pase_review(review)
        advantages.append(advnt)
        disadvantages.append(dsadvnt)
        commtents.append(cmnt)
    united_advantages = unite_same_type(advantages, fragment_length_limit=FRAGMENT_LENGHT_LIMIT, separator='\n|\n')
    united_disadvantages = unite_same_type(disadvantages, fragment_length_limit=FRAGMENT_LENGHT_LIMIT, separator='\n|\n')
    united_commtents = unite_same_type(commtents, fragment_length_limit=FRAGMENT_LENGHT_LIMIT, separator='\n|\n')
    return united_advantages, united_disadvantages, united_commtents

class ReviewSummary(BaseModel):
    advantages: list = Field(..., description="list of main advantages")
    # disadvantages: list = Field(..., description="list of main disadvantages")
    # comments: list = Field(..., description="overall comments")

# assert os.path.exists('/home/amstel/llm/models/mixtral-8x7b-moe-rp-story.Q4_K_M.gguf')

chat_model = LlamaCpp(
    # model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct.Q6_K.gguf',
    # model_path='/home/amstel/llm/models/Publisher/Repository/Llama-3-8B-Instruct-32k-v0.1.Q6_K.gguf',
    # model_path='/home/amstel/llm/models/Publisher/Repository/llama-3-8b-instruct-262k.Q6_K.gguf',
    model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf',

    # model_path = '/home/amstel/llm/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf',
    # model_path = '/home/amstel/llm/models/mixtral-8x7b-moe-rp-story.Q4_K_M.gguf',
    n_gpu_layers=33, # 28 for llama2 13b, 10 for mixtral
    max_tokens=1024,
    n_batch=1024,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    temperature=0.0,
    # repeat_penalty=1.5,
    # grammar_path='/home/amstel/llm/src/grammars/interface_json_arr.gbnf',
    stop=["<|eot_id|>", ],
)

# todo: summarize each review separately

def def_debugger(inp):
    logger.info(inp)
    return inp



debugger = RunnablePassthrough(def_debugger)
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

template_a = PromptTemplate.from_template(template=llama_raw_template_system_a_en+llama_raw_template_user)
template_d = PromptTemplate.from_template(template=llama_raw_template_system_d_en+llama_raw_template_user)
template_c = PromptTemplate.from_template(template=llama_raw_template_system_c_en+llama_raw_template_user)

# def preprocessor(d:dict) -> str:
#     # logger.debug(d)
#     s = d.get('context')
#     logger.debug(len(s))
#     # s = s[:2050]
#     # logger.debug(s)
#     return s

# context_preprocessor = RunnableLambda(preprocessor)
json_parser = JsonOutputParser()
str_parser = StrOutputParser()
chain_advantages = template_a | chat_model | json_parser
chain_disadvantages = template_d | chat_model | json_parser
chain_comments = template_c | chat_model | json_parser

if __name__ == '__main__':
    con_product_reviews = MongoConnector(operation='read', db_name='scraped_data', collection_name='product_reviews')
    cursor_product_reviews = con_product_reviews.read_many({})
    product_reviews = list(cursor_product_reviews)

    con_product_review_summarizations = MongoConnector(operation='read', db_name='scraped_data', collection_name='product_review_summarizations')
    cursor_product_review_summarizations = con_product_review_summarizations.read_many({})
    product_review_summarizations = list(cursor_product_review_summarizations)

    # deduct already summarized products from the cursor_product_reviews
    already_summarized_names = []
    for review in product_review_summarizations:
        already_summarized_names.extend([key for key in review.keys() if key != '_id'])

    candidates_to_summarize_names = []
    for review in product_reviews:
        candidates_to_summarize_names.extend([key for key in review.keys() if key != '_id'])

    yet_to_summarize = set(candidates_to_summarize_names) - set(already_summarized_names)
    # product_reviews = [review for review in product_reviews if len(set(yet_to_summarize) & set(review.keys())) > 0]  # write only new entries
    output_dict = {}  # name -> List[review_description]
    for i, doc in enumerate(product_reviews):
        review_descriptions = []
        keys = list(doc.keys())
        assert len(keys) == 2
        key = keys[-1]
        assert key != '_id'
        reviews_list = doc[key]
        for review in reviews_list:
            review_description = review.get('review_description')
            if review_description:
                review_descriptions.append(review_description)
        # print(len(review_descriptions))
        output_dict[key] = review_descriptions


    con = MongoConnector(operation='write', db_name='scraped_data', collection_name='product_review_summarizations')
    summarized_reviews = {}
    # logger.debug(len(output_dict))
    i = 0
    for model_name, reviews_body in output_dict.items():
        # reviews_body is list of up to 10 reviews of one product
        united_advantages_list, united_disadvantages_list, united_comments_list = get_united_types(reviews_body, )
        # united_advantages = strip_latin(united_advantages)
        summarized_advantages = {'advantages': []}
        summarized_disadvantages = {'disadvantages': []}
        for ua in united_advantages_list:
            try:
                summarized_data = chain_advantages.invoke(input={'context': ua}, config={"callbacks": [langfuse_callback_handler]})
                # logger.info(summarized_data)
                summarized_advantages['advantages'] += summarized_data.get('advantages')
            except Exception as e:
                logger.critical(f"{model_name}--advantages: {e}")
        for ud in united_disadvantages_list:
            try:
                summarized_data = chain_disadvantages.invoke(input={'context': ud}, config={"callbacks": [langfuse_callback_handler]})
                # logger.warning(summarized_data)
                summarized_disadvantages['disadvantages'] += summarized_data.get('disadvantages')
            except Exception as e:
                logger.critical(f"{model_name}--disadvantages: {e}")
        logger.info(f'current position: {i}')
        i += 1


        logger.info(summarized_advantages)
        logger.warning(summarized_disadvantages)

        # print(type(summarized_json))
        # todo
        summarized_json = {
            'advantages': summarized_advantages,
            'disadvantages': summarized_disadvantages,
            'inserted_datetime': datetime.now(),
            'llm_name': 'llama3_v2',
        }
        # logger.warning(summarized_json)
        # con.write_one({model_name: summarized_json})
        summarized_reviews[model_name] = summarized_json

    logger.info(summarized_reviews)
    with open('/home/amstel/llm/out/summarized_reviews.pkl', 'wb') as f:
        pickle.dump(summarized_reviews, f)

    print('success')