from langchain_community.llms import LlamaCpp
from langchain_community.utilities import SQLDatabase
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.llms import LlamaCpp
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
import os
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from datetime import datetime

trace_name = f'sql_llama3_{datetime.now()}'
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-a45f2a3d-4085-4170-b337-8cc2f1921aef"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-26acbd19-74af-4cb4-93e1-919526c13921"

langfuse = Langfuse()
langfuse_callback_handler = CallbackHandler(trace_name=trace_name)
db = SQLDatabase.from_uri("postgresql://localhost:6432/scraperdb?user=scraperuser&password=scraperpassword")
logger.warning(f'db type: {type(db)}')

top_k = 1
llm = LlamaCpp(
    # model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf',
    model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct.Q6_K.gguf',
    # model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q8_0.gguf',
    # model_path='/home/amstel/llm/models/Publisher/Repository/meta-llama-3-8b-instruct.fp16.gguf',
    n_gpu_layers=33,
    max_tokens=256,
    n_batch=128,
    n_ctx=2048,
    f16_kv=True,
    verbose=False,
    temperature=0.0,
    stop=['<|eot_id|>', '```', '```\n', ],
)

dialect = 'postgresql'

table_info = '''create table scraped_data.washing_machine (
  "brand" text, -- название производителя
  "rating_value" real, -- рейтинг товара
  "rating_count" real, -- количество оценок
  "review_count" real, -- количество отзывов
  "name" text, -- название товара
  "price" real, -- цена, руб.
  "max_load" real, -- максимальная загрузка, кг.
  "depth" real, -- глубина, см.
  "drying" text -- есть ли сушка, Да/Нет
)

/*
3 rows from scraped_data.washing_machine table:
brand|rating_value|rating_count|review_count|name|price|max_load|depth|drying
Candy|4.5|966|299|Стиральная машина Candy AQUA 114D2-07|882.99|4|43|Нет
LG|5|6|2|Стиральная машина LG F2J3WS2W|1273.91|6.5|44|Да
Indesit|4.5|921|290|Стиральная машина Indesit IWSB 51051 BY|120|5|40|Нет
*/'''

examples = [
    {"input": "Фирма Электролюкс без сушки глубина до 50 см", "query": "SELECT * FROM scraped_data.washing_machine WHERE brand ILIKE '%Electrolux%' AND (drying = 'Нет' or drying is null) AND depth <= 50;"},
    {"input": "популярная, хорошая, недорогая", "query": "SELECT * FROM scraped_data.washing_machine WHERE price <= 1000;"},
    {"input": "Cтиральная машина с сушкой", "query": "SELECT * FROM scraped_data.washing_machine WHERE drying = 'Да';"},
    {"input": "отличная стиралка", "query": "SELECT * FROM scraped_data.washing_machine WHERE rating_value >= 4.8;"},
    {"input": "Бош, от 5 кг, хорошая лучшая", "query": "SELECT * FROM scraped_data.washing_machine WHERE brand ILIKE '%Bosch%' AND max_load >= 5 and rating_value >= 4.8;"},
    {"input": "хорошая стиральная машина",   "query": "SELECT * FROM scraped_data.washing_machine WHERE rating_value >= 4.5;"},
    {"input": "дешевая, с отзывами", "query": "SELECT * FROM scraped_data.washing_machine WHERE price <= 1000;"},
    {"input": "Производители Атлант, Хаер, Ханса, Самсунг", "query": "SELECT * FROM scraped_data.washing_machine WHERE brand ILIKE ANY(ARRAY['%Atlant%','%Haier%','%Hansa%','%Samsung%']);"},
    {"input": "лучшие стиральные машины", "query": "SELECT * FROM scraped_data.washing_machine;"},
]

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

system_message_llama = (
    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n'
    'You are an postgresql expert. Given an input question, create a syntactically correct postgresql query to run. '\
    ' Here are the requirements you must follow:\n'
    ' translate "brand" into latin characters;\n'
    ' if needed to filter by fields "brand", "name" you MUST use operator "ILIKE" with "%" instead of "=";\n'
    ' filter by field "brand" only if "brand" is mentioned in "User input";\n\n' 
    'Here is the relevant table info: \n{table_info}\n\n'
)
system_message_mixtral = "You are a postgres expert. Given an input question, create a syntactically correct postgres query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."

prompt = FewShotPromptTemplate(
    examples=examples[:20],
    example_prompt=example_prompt,
    prefix=system_message_llama,
    suffix="<|eot_id|><|start_header_id|>user<|end_header_id|>\nUser input: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n```sql",
    input_variables=["input", "top_k", "table_info"],
)

chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "input": "стиральная машина Хаер",
    "table_info": table_info,
    "top_k":top_k,
    }, config={"callbacks": [langfuse_callback_handler],})
logger.warning(response)