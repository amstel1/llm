# raise SystemError

import sys
sys.path.append('/home/amstel/llm')
import pickle
from src.mongodb.mongo_utils import MongoConnector
from loguru import logger
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import os
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from datetime import datetime
trace_name = f'summary_llama3_{datetime.now()}'
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-a45f2a3d-4085-4170-b337-8cc2f1921aef"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-26acbd19-74af-4cb4-93e1-919526c13921"
langfuse = Langfuse()
langfuse_callback_handler = CallbackHandler(trace_name=trace_name)

import json
from typing import List
import sys
sys.path.append('/home/amstel/llm/src')
from mongodb.mongo_utils import MongoConnector
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import AIMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# second best -- looks like this may be the better prompt
mistral_raw_template_system = '<s> [INST] \n Ты эксперт мирового уровня по анализу отзывов на товары. Выдели главные преимущества и недостатки товара из отзывов ниже. Отвечай только на русском языке. Ты всегда должен сформировать ответ в виде JSON с ключами "advantages" и "disadvantages".\n [/INST]' \

# best - idk
# mistral_raw_template_system = '<s> [INST] Ты эксперт мирового уровня по анализу отзывов на товары. В качестве ответа ты должен вернуть только JSON с ровно двумя ключами: "достоинства", "недостатки". Значения должны содержать список достоинств и недостатков данного товара, которые указаны в отзывах. [/INST] </s> \n\n'
# mistral_raw_template_context = ("<s> [INST] Отзывы: {context}. [/INST] </s> \n\n")

mistral_raw_template_context = "<s> [INST] \n Отзывы: {context}.\n [/INST]" # v3

llama_raw_template_system_ru = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты эксперт мирового уровня по анализу отзывов на товары. Выдели главные преимущества и недостатки товара из отзывов ниже. Отвечай ТОЛЬКО на русском языке. Ты всегда должен сформировать ответ в виде JSON с ключами "advantages" и "disadvantages".\n\nОтзывы:\n\n\n"""

llama_raw_template_system_en = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a world class expert in summarizing reviews. Extract main advantages and disadvantages of the consumer goods described below. Respond only in Russian. You must return a valid JSON with keys "advantages" и "disadvantages".\n\nReviews:\n\n\n"""


llama_raw_template_user = """{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nJSON: """

class ReviewSummary(BaseModel):
    advantages: list = Field(description="list of main advantages")
    disadvantages: str = Field(description="list of main disadvantages")
    # comments: str = Field(description="overall conclusion")

# assert os.path.exists('/home/amstel/llm/models/mixtral-8x7b-moe-rp-story.Q4_K_M.gguf')


chat_model = LlamaCpp(
    model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct.Q6_K.gguf',
    # model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf',





    # model_path = '/home/amstel/llm/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf',
    # model_path = '/home/amstel/llm/models/mixtral-8x7b-moe-rp-story.Q4_K_M.gguf',
    n_gpu_layers=33, # 28 for llama2 13b, 10 for mixtral
    max_tokens=512,
    n_batch=128,
    n_ctx=8192,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    temperature=0.0,
    # repeat_penalty=1.5,
    stop=["<|eot_id|>", "<|start_header_id|>"],
)

# todo: summarize each review separately

def def_debugger(inp):
    logger.info(inp)
    return inp



debugger = RunnablePassthrough(def_debugger)

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


template = PromptTemplate.from_template(template=llama_raw_template_system_ru+llama_raw_template_user)

def preprocessor(d:dict) -> str:
    # logger.debug(d)
    s = d.get('context')
    logger.debug(len(s))
    s = s.replace('\n', '')[:20000]
    logger.debug(s)
    return s

context_preprocessor = RunnableLambda(preprocessor)
parser = JsonOutputParser(pydantic_object=ReviewSummary)
chain = (
        {"context": context_preprocessor} |
         template |
         # debugger |
         chat_model |
         parser
        )

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
        print(len(review_descriptions))
        output_dict[key] = review_descriptions


    con = MongoConnector(operation='write', db_name='scraped_data', collection_name='product_review_summarizations')
    summarized_reviews = {}
    # logger.debug(len(output_dict))
    for model_name, reviews_body in output_dict.items():
        ctx = '\n\n'.join(reviews_body)
        print(len(ctx))
        logger.info(f'{model_name}: {len(reviews_body)}')
        summarized_json = chain.invoke(input={'context': ctx,}, config={"callbacks": [langfuse_callback_handler],})  # llm call output
        print(type(summarized_json))
        summarized_json.update({'inserted_datetime': datetime.now(), 'llm_name': 'llama3'})
        logger.warning(summarized_json)
        # con.write_one({model_name: summarized_json})
        summarized_reviews[model_name] = summarized_json
        break

    with open('/home/amstel/llm/out/summarized_reviews.pkl', 'wb') as f:
        pickle.dump(summarized_reviews, f)

    print('success')