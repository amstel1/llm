from langchain_community.llms import Ollama
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
from langchain_community.agent_toolkits import create_sql_agent
import pandas as pd
import sys
sys.path.append('/home/amstel/llm/src')
from postgres.config import user, password, host, port, database
from typing import List
from general_llm.llm_endpoint import call_generate_from_query_api, call_generation_api, call_generate_from_history_api
from sqlalchemy.sql import text


class SqlToText:
    @classmethod
    def sql_query(cls, user_query: str) -> pd.DataFrame:

        # trace_name = f'sql_llama3_{datetime.now()}'
        # os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
        # os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-8c20497b-23da-4267-961c-f66e33a8bee4"
        # os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-641f1b97-5f99-456a-8597-a44a8f7fc6ab"
        # langfuse = Langfuse()
        # langfuse_callback_handler = CallbackHandler(trace_name=trace_name)

        # llm = Ollama(model=self.ollama_model, stop=self.ollama_stop, num_gpu=-1, num_thread=-1, temperature=0,
        #              mirostat=0)
        # llm = LlamaCpp(
        #     model_path='/home/amstel/llm/models/Publisher/Repository/Meta-Llama-3-8B-Instruct-Q6_K.gguf',
        #     n_gpu_layers=33,
        #     max_tokens=-1,
        #     n_batch=512,
        #     n_ctx=2048,
        #     f16_kv=False,
        #     verbose=True,
        #     temperature=0.0,
        #     stop=self.ollama_stop,
        # )
        top_k = 1
        dialect = 'postgresql'
        uri = f"postgresql://{host}:{port}/{database}?user={user}&password={password}"
        db = SQLDatabase.from_uri(uri)
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
            {"input": "Фирма Электролюкс без сушки глубина до 50 см",
             "query": "SELECT name, price, rating_value, drying, depth FROM scraped_data.washing_machine WHERE brand ILIKE '%Electrolux%' AND (drying = 'Нет' or drying is null) AND depth <= 50;"},
            {"input": "популярная, хорошая, недорогая",
             "query": "SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE price <= 1000;"},
            {"input": "Cтиральная машина с сушкой",
             "query": "SELECT name, price, rating_value, drying FROM scraped_data.washing_machine WHERE drying = 'Да';"},
            {"input": "отличная стиралка",
             "query": "SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE rating_value >= 4.8;"},
            {"input": "Бош, от 5 кг, хорошая лучшая",
             "query": "SELECT name, price, rating_value, max_load FROM scraped_data.washing_machine WHERE brand ILIKE '%Bosch%' AND max_load >= 5 and rating_value >= 4.8;"},
            {"input": "хорошая стиральная машина",
             "query": "SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE rating_value >= 4.5;"},
            {"input": "дешевая",
             "query": "SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE price <= 1000;"},
            {"input": "Производители Атлант, Хаер, Ханса, Самсунг",
             "query": "SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE brand ILIKE ANY(ARRAY['%Atlant%','%Haier%','%Hansa%','%Samsung%']);"},
        ]

        example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

        system_message_llama = (
            '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n'
            'You are a postgresql expert. Given an input question, create a syntactically correct postgresql query to run. ' \
            ' Here are the requirements you must follow:\n'
            ' translate "brand" into latin characters;\n'
            ' if needed to filter by fields "brand", "name" you MUST use operator "ILIKE" with "%" instead of "=";\n'
            ' filter by field "brand" only if "brand" is mentioned in "User input";\n\n'
            'Here is the relevant table info: \n{table_info}\n\n'
        )

        # prompt = FewShotPromptTemplate(
        #     examples=examples[:20],
        #     example_prompt=example_prompt,
        #     prefix=system_message_llama,
        #     suffix="<|eot_id|><|start_header_id|>user<|end_header_id|>\nUser input: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n```sql",
        #     input_variables=["input", "top_k", "table_info"],
        # )
        # if hasattr(prompt, 'format'):
        #     prompt_string = prompt.format(
        #         input=user_input,
        #         table_info=table_info,
        #         top_k=top_k
        #     )

        # def _debugger(x):
        #     logger.info(x)
        #     return x

        # logger.critical(type(prompt_string))
        # logger.debug(prompt)
        # chain = prompt | RunnablePassthrough(_debugger) | llm | StrOutputParser()
        str_prompt = (
'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n'\
'You are a top class business analyst that specializes in translating natural language queries into SQL. Perform the task you are assigned to to the best of your ability. <|eot_id|><|start_header_id|>user<|end_header_id|>\n'\
f"""Given an input_query, create a valid SQL query to run. 


Here is the relevant table info:  
create table scraped_data.washing_machine (
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


Here are the rows from scraped_data.washing_machine table: 
brand | rating_value | rating_count | review_count | name | price | max_load | depth | drying 
Candy | 4.5 | 966 | 299 | Стиральная машина Candy AQUA 114D2-07 | 882.99 | 4 | 43 | Нет 
LG | 5 | 6 | 2 |Стиральная машина LG F2J3WS2W | 1273.91 | 6.5 | 44 | Да 
Indesit | 4.5 | 921 | 290 | Стиральная машина Indesit IWSB 51051 BY | 120 | 5 | 40 |Нет 


Here are the examples of correct pairs of input_query (Q) and required output (SQL):

Q: Фирма Электролюкс без сушки глубина до 50 см;
SQL: SELECT name, price, rating_value, drying, depth FROM scraped_data.washing_machine WHERE (brand ILIKE '%Electrolux%') AND (drying = 'Нет' or drying is null) AND depth <= 50;

Q: популярная, хорошая, недорогая
SQL: SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE price <= 1000;

Q: Cтиральная машина с сушкой
SQL: SELECT name, price, rating_value, drying FROM scraped_data.washing_machine WHERE drying = 'Да';

Q: отличная стиралка;
SQL: SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE rating_value >= 4.8;

Q: от 5 кг, хорошая лучшая
SQL: SELECT name, price, rating_value, max_load FROM scraped_data.washing_machine WHERE (max_load >= 5) and (rating_value >= 4.8);

Q: хорошая стиральная машина
SQL: SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE rating_value >= 4.5;

Q: дешевая
SQL: SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE price <= 1000;


input_query: {user_query}
SQL:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n```sql""")

        logger.warning('str_prompt')
        logger.warning(str_prompt)
        #api call from string
        response = call_generation_api(prompt=str_prompt, grammar=None, stop=['<|eot_id|>', '```', '```\n',])
        response_query = text(response.strip().replace('\n', ' '))

        # response_query = chain.invoke({
        #     "input": user_input,
        #     "table_info": table_info,
        #     "top_k": top_k,
        # }, config={"callbacks": [
        #     # langfuse_callback_handler
        # ], })
        logger.warning(response_query)

        df = pd.read_sql(
            sql=response_query,
            con=uri,
        )
        if 'price' in df.columns: df = df[df.price.notnull()]
        if 'rating_value' in df.columns: df = df[df.rating_value.notnull()]
        if 'name' in df.columns:
            if df['name'].nunique() != df.shape[0]:
                if 'price' in df.columns:
                    df.sort_values(['price', 'name'], inplace=True)
                if 'rating_value' in df.columns:
                    df.sort_values(["rating_value", 'price', 'name'], ascending=[False, False, True], inplace=True)
                    df.drop_duplicates(subset=['name'], keep='first', inplace=True)
        logger.warning(f'{df.shape}')
        logger.warning(f'{df.head()}')
        return df


if __name__ == '__main__':
    # user_query = "стиральная машина глубина до 43, загрузка от 6, с рейтингом от 4.8, производитель Korting"
    # user_query = "дешевая стиральная машина"
    # user_query = 'Поможете найти недорогую стиральную машину, которая работает хорошо?'
    # user_query = 'Недорогая стиральная машина с хорошими характеристиками.'
    user_query = "Суть запроса: ищется стиральная машина от производителей Electrolux, Samsung, LG, Bosch, с загрузкой от 6 кг, глубиной до 43 см и ценой до 2000 рублей, с любым рейтингом"
    # user_query = "Суть требований пользователя: стиральная машина с хорошим брендом, узкой и вместительной."
    response = SqlToText.sql_query(user_query=user_query)
    print(response)


