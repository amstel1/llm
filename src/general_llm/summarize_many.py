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
import json
import sys
sys.path.append('/home/amstel/llm/src')
from mongodb.mongo_utils import MongoConnector
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from datetime import datetime
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import AIMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate



mistral_raw_template_system = '<s> [INST] You are a world class review analyst. You must in great detail summarize advantages and disadvantages from the consumer goods reviews below. The output JSON keys must ONLY be both "advantages" and "disadvantages". [/INST]\n\n'
mistral_raw_template_context = " [INST] Reviews: {context} [/INST]\n\n"

class ReviewSummary(BaseModel):
    advantages: list = Field(description="list of main advantages")
    disadvantages: str = Field(description="list of main disadvantages")
    # comments: str = Field(description="overall conclusion")

# assert os.path.exists('/home/amstel/llm/models/mixtral-8x7b-moe-rp-story.Q4_K_M.gguf')


chat_model = LlamaCpp(
    model_path = '/home/amstel/llm/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf',
    # model_path = '/home/amstel/llm/models/mixtral-8x7b-moe-rp-story.Q4_K_M.gguf',
    n_gpu_layers=33, # 28 for llama2 13b, 10 for mixtral
    max_tokens=1000,
    n_batch=256,
    n_ctx=6044,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    temperature=0.0,
    # repeat_penalty=0.01,
    stop=["</s>"],
)


def def_debugger(inp):
    logger.info(inp)
    return inp



debugger = RunnablePassthrough(def_debugger)

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=mistral_raw_template_system),
        HumanMessagePromptTemplate.from_template(template=mistral_raw_template_context),
        AIMessage(content=' JSON: '),
    ]
)

def preprocessor(d:dict) -> str:
    # logger.debug(d)
    s = d.get('context')
    # logger.debug(s)
    return s.replace('\n', '')
context_preprocessor = RunnableLambda(preprocessor)

parser = JsonOutputParser(pydantic_object=ReviewSummary)
chain = (
        # {"context": context_preprocessor} |
         template |
         # debugger |
         chat_model |
         parser
        )

if __name__ == '__main__':
    con = MongoConnector(operation='read', db_name='scraped_data', collection_name='product_reviews')
    cursor = con.read_many({})
    output_dict = {}  # name -> List[review_description]
    for i, doc in enumerate(cursor):
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
        ctx = '\n\n\n'.join(reviews_body)
        logger.info(f'{model_name}: {len(reviews_body)}')
        summarized_json = chain.invoke({'context': ctx,}) # llm call output
        # print(type(summarized_json))
        summarized_json.update({'inserted_datetime': datetime.now(), 'llm_name':'mistral_7b'})
        logger.warning(summarized_json)
        con.write_one({model_name: summarized_json})
        summarized_reviews[model_name] = summarized_json
        # break

    with open('/home/amstel/llm/out/summarized_reviews.pkl', 'wb') as f:
        pickle.dump(summarized_reviews, f)

    print('success')