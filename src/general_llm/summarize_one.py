import sys
sys.path.append('/home/amstel/llm')
sys.path.append('/home/amstel/llm/src')

# prod_name = 'Стиральная машина LG F2V5GS0W'
# prod_name = 'Стиральная машина BEKO WSPE7H616W'
# prod_name = "Стиральная машина LG F2V3GS6W"
# prod_name = "Стиральная машина Renova WS-30ET"
# prod_name = "Стиральная машина BEKO WRE 6512 BWW (BY)"
prod_name = 'Стиральная машина ATLANT СМА 80С1214-01'
import sys

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, ChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from src.mongodb.mongo_utils import MongoConnector


mc = MongoConnector('read', 'scraped_data', 'product_reviews')
resp = mc.read_one({prod_name:{'$exists': True}})
descriptions_list = [x['review_description'] for x in resp[prod_name]]
context = '\n'.join(descriptions_list)

class ReviewSummary(BaseModel):
    advantages: list = Field(description="list of main advantages")
    disadvantages: str = Field(description="list of main disadvantages")
    # comments: str = Field(description="overall conclusion")

from loguru import logger
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
def def_debugger(inp):
    logger.info(inp)
    return inp


# mixtral_raw_template_system = '<s> [INST] <<SYS>>\n You are a world class review analyst. Summarize advantages and disadvantages from the consumer goods reviews below. Output must be a well-formed JSON object with keys must only in ("advantages", "disadvantages".\n <</SYS>> [/INST]'
# mixtral_raw_template_context = f" [INST] Reviews: {context} [/INST]\n\n"

# mistral_raw_template_system = '<s> [INST] \n Ты эксперт мирового уровня по анализу отзывов на товары. Выдели главные преимущества и недостатки товара из отзывов ниже. Отвечай только на русском языке. Ты всегда должен сформировать ответ в виде JSON с ключами "advantages" и "disadvantages".\n [/INST]' \
# mistral_raw_template_system = '<s> [INST] Ты эксперт мирового уровня по анализу отзывов на товары. В формате JSON верни главные достоинства и недостатки товара. JSON должен иметь ровно два ключа: "достоинства", "недостатки". Значения должны содержать список достоинств и недостатков данного товара, которые указаны в отзывах. [/INST] </s> \n\n'

# best
# mistral_raw_template_system = '<s> [INST] Ты эксперт мирового уровня по анализу отзывов на товары. В качестве ответа ты должен вернуть только JSON с ровно двумя ключами: "достоинства", "недостатки". Значения должны содержать список достоинств и недостатков данного товара, которые указаны в отзывах. [/INST] </s> \n\n'

# raw
# mistral_raw_template_context = (f"<s> [INST] Отзывы: {context}. [/INST] </s> \n\n")

debugger = RunnablePassthrough(def_debugger)

chat_model = LlamaCpp(
    # llama 13b saiga: -- '../models/model-q4_K(2).gguf'
    # roleplay - mixtral moe 8x7b: -- mixtral-8x7b-moe-rp-story.Q4_K_M
    # model_path='/home/amstel/llm/models/mixtral-8x7b-v0.1.Q4_K_M.gguf',
    # model_path='/home/amstel/llm/models/qwen1_5-14b-chat-q4_k_m.gguf',

    # model_path='/home/amstel/llm/models/mixtral-8x7b-moe-rp-story.Q4_K_M.gguf',
    # model_path = '/home/amstel/llm/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf',

    model_path = '/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct.Q6_K.gguf',
    n_gpu_layers=33,
    max_tokens=200,
    n_batch=1024,
    n_ctx=6000,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    temperature=0.0,
    # stop=["</s>"],
)

llama_raw_template_system = """"""
llama_raw_template_user = """"""

template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=llama_raw_template_system),
        HumanMessage(content=llama_raw_template_user),
        AIMessage(content=' JSON:'),
    ]
)

parser = JsonOutputParser()
chain = (template |
         debugger |
         chat_model |
         parser
         )

print(chain.invoke(input={}))
