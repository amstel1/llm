# summarize reviews

context = '''Отдельно стоящая, барабанного типа, глубина 41.5 см, загрузка фронтальная, 6 кг, количество программ 15, класс энергопотребления А++, материал бака пластик, отложенный старт, обработка паром, индикация ошибок, звуковой сигнал, защита от детей, контроль дисбаланса, контроль пенообразования, ширина 60 см.'''

# raw_template = f'''<s> [INST] Ты суммаризатор достоинств и недостатков товаров. На основе контекста ниже сделай список плюс и минусов товара. [/INST]
# Контекст: {context}
# Ответ: </s>
# '''
raw_template = f'''<s> [INST] Below is the list of features of a consumer product separated by a ",". You need to create a database field name for each feature. Return a JSON with key and value pairs. [/INST]
Feature: {context}
JSON: 
'''

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


# class ReviewSummary(BaseModel):
#     advantages: str = Field(description="adavantages of the product")
#     disadvanatges: str = Field(description="disadavantages of the product")


chat_model = LlamaCpp(
    # llama 13b saiga: -- '../models/model-q4_K(2).gguf'
    # roleplay - mixtral moe 8x7b: -- mixtral-8x7b-moe-rp-story.Q4_K_M
    # mixtral-8x7b-v0.1.Q4_K_M
    model_path='/home/amstel/llm/models/mixtral-8x7b-moe-rp-story.Q4_K_M.gguf',
    n_gpu_layers=8,  # 28 for llama2 13b, 10 for mixtral
    max_tokens=1000,
    n_batch=128,
    n_ctx=4096,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    temperature=0.0,
)

template = ChatPromptTemplate.from_template(raw_template)
# parser = JsonOutputParser(pydantic_object=ReviewSummary)
chain = template | chat_model #| parser

print(chain.invoke(input={}))
