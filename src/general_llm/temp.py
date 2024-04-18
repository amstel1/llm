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
import sys
sys.path.append('/home/amstel/llm/src')
from mongodb.mongo_utils import MongoConnector
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

# raw_template = f"<s>[INST] You summarize advantages and disadvantages of the consumer goods. Based on the reviews below respond with a JSON of pros and cons of the product. Make sure not to repeat yourself. [/INST]" \
# "/n Reviews: {ctx}" \
# "/n JSON:"


# raw_template = '''Ты суммаризатор достоинств и недостатков товаров. На основе контекста ниже сделай JSON из достоинств и недостатков товара. В ответе используй ТОЛЬКО русский язык.
# Контекст: {ctx}
# JSON:
# '''

raw_template_system = """Ты суммаризатор достоинств и недостатков товаров. На основе контекста ниже сформируй JSON из достоинств и недостатков товара. В JSON используй только грамматически верный русский язык.


Контекст: {context}


"""
print(raw_template_system)

class ReviewSummary(BaseModel):
    достоинства: str = Field(description="достоинства товара")
    недостатки: str = Field(description="недостатки товара")
    комментарий: str = Field(description="вывод о товаре")

# assert os.path.exists('/home/amstel/llm/models/mixtral-8x7b-moe-rp-story.Q4_K_M.gguf')

chat_model = LlamaCpp(
    # llama 13b saiga: -- '../models/model-q4_K(2).gguf'
    # roleplay - mixtral moe 8x7b: -- mixtral-8x7b-moe-rp-story.Q4_K_M
    # mixtral-8x7b-v0.1.Q4_K_M
    model_path='/home/amstel/llm/models/qwen1_5-14b-chat-q4_k_m.gguf',
    n_gpu_layers=20,  # 28 for llama2 13b, 10 for mixtral
    max_tokens=1000,
    n_batch=128,
    n_ctx=6044,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    temperature=0.0,
)

def def_debugger(inp):
    logger.info(inp)
    return inp

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import AIMessagePromptTemplate, SystemMessagePromptTemplate

debugger = RunnablePassthrough(def_debugger)
template = ChatPromptTemplate.from_messages(
    messages = [
        # SystemMessage(content=raw_template_system),
        SystemMessagePromptTemplate.from_template(raw_template_system),
        AIMessage(content='JSON: '),
    ]
)

# template = ChatPromptTemplate.from_messages(
#     messages=[
#         {'role': 'system', 'content':raw_template_system},
#         {'role': 'assistant', 'content': 'JSON: '},
#     ], input_variables=['ctx'],
# )

# template = PromptTemplate(template=raw_template, input_variables=['ctx'])
parser = JsonOutputParser(pydantic_object=ReviewSummary)
chain = template | debugger | chat_model | parser

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
        break

    con = MongoConnector(operation='write', db_name='scraped_data', collection_name='product_review_summarizations')
    summarized_reviews = {}
    for k, v in output_dict.items():
        ctx = '\n\n'.join(v)
        logger.info(f'{k}: {len(v)}')
        summarized_json = chain.invoke({'context': ctx})  # llm call output
        logger.warning(summarized_json)
        con.write_one({k: summarized_json})
        summarized_reviews[k] = summarized_json
        break

    with open('../out/summarized_reviews.pkl', 'wb') as f:
        pickle.dump(summarized_reviews, f)

    print('success')