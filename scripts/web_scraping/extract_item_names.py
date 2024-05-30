url = 'https://catalog.onliner.by/mobile'
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel
from loguru import logger

chat_model = LlamaCpp(
    # llama 13b saiga: -- '../models/model-q4_K(2).gguf'
    # roleplay - mixtral moe 8x7b: -- mixtral-8x7b-moe-rp-story.Q4_K_M
    # mixtral-8x7b-v0.1.Q4_K_M
    model_path='/home/amstel/llm/models/saiga_mistral_7b.gguf',
    n_gpu_layers=10,  # 28 for llama2 13b, 10 for mixtral
    max_tokens=500,
    n_batch=32,
    n_ctx=1024,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,
    temperature=0.0,
)
# context = 'Смартфон Xiaomi 14 12GB/512GB международная версия (матовый черный)'
context = 'Смартфон Xiaomi 14 12GB/512GB международная версия (серебристо-белый)'


system_template = SystemMessagePromptTemplate.from_template("<s>[INST] You are a world class proper name extractor. Return the only one proper name from the context: [/INST]</s>")
context_template = HumanMessagePromptTemplate.from_template("{context}")
template = ChatPromptTemplate.from_messages(messages=[system_template, context_template],)
# messages = template.format_messages(context=context)
chat = template | chat_model
response = chat.invoke({'context': context})
logger.info(response)