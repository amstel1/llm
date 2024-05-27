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

class SqlToText:
    def __init__(self,
                 ollama_model: str = "llama3_q6_32k",
                 ollama_stop: List[str] = ['<|eot_id|>', '```', '```\n', ]
                 ):
        self.ollama_model = ollama_model
        self.ollama_stop = ollama_stop

    def sql_query(self, user_input: str) -> pd.DataFrame:

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

        prompt = FewShotPromptTemplate(
            examples=examples[:20],
            example_prompt=example_prompt,
            prefix=system_message_llama,
            suffix="<|eot_id|><|start_header_id|>user<|end_header_id|>\nUser input: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n```sql",
            input_variables=["input", "top_k", "table_info"],
        )
        if hasattr(prompt, 'format'):
            prompt_string = prompt.format(
                input=user_input,
                table_info=table_info,
                top_k=top_k
            )

        def _debugger(x):
            logger.info(x)
            return x

        logger.critical(type(prompt_string))
        # logger.debug(prompt)
        # chain = prompt | RunnablePassthrough(_debugger) | llm | StrOutputParser()
        str_prompt = (f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n'
                      f'You are a postgresql expert. Given an input question, create a syntactically correct postgresql query to run.'
                      f'Here are the requirements you must follow:'
                      f' translate "brand" into latin characters;'
                      f' if needed to filter by fields "brand", "name" you MUST use operator "ILIKE" with "%" instead of "=";'
                      f' filter by field "brand" only if "brand" is mentioned in "User input"<|eot_id|><|start_header_id|>user<|end_header_id|>\n'
                      f' Here is the relevant table info: '
                      f' create table scraped_data.washing_machine ('
                      f' "brand" text, -- название производителя'
                      f' "rating_value" real, -- рейтинг товара'
                      f' "rating_count" real, -- количество оценок'
                      f' "review_count" real, -- количество отзывов'
                      f' "name" text, -- название товара'
                      f' "price" real, -- цена, руб.'
                      f' "max_load" real, -- максимальная загрузка, кг.'
                      f' "depth" real, -- глубина, см.'
                      f' "drying" text -- есть ли сушка, Да/Нет'
                      f' )'
                      f' /*'
                      f' 3 rows from scraped_data.washing_machine table:'
                      f' brand|rating_value|rating_count|review_count|name|price|max_load|depth|drying'
                      f' Candy|4.5|966|299|Стиральная машина Candy AQUA 114D2-07|882.99|4|43|Нет'
                      f' LG|5|6|2|Стиральная машина LG F2J3WS2W|1273.91|6.5|44|Да'
                      f' Indesit|4.5|921|290|Стиральная машина Indesit IWSB 51051 BY|120|5|40|Нет'
                      f' */'
                      f'\n\nUser input: Фирма Электролюкс без сушки глубина до 50 см;\nSQL query: SELECT name, price, rating_value, drying, depth FROM scraped_data.washing_machine WHERE brand ILIKE \'%Electrolux%\' AND (drying = \'Нет\' or drying is null) AND depth <= 50;'
                      f'\n\nUser input: популярная, хорошая, недорогая\nSQL query: SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE price <= 1000;'
                      f'\n\nUser input: Cтиральная машина с сушкой\nSQL query: SELECT name, price, rating_value, drying FROM scraped_data.washing_machine WHERE drying = \'Да\';'
                      f'\n\nUser input: отличная стиралка;\nSQL query: SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE rating_value >= 4.8;'
                      f'\n\nUser input: Бош, от 5 кг, хорошая лучшая\nSQL query: SELECT name, price, rating_value, max_load FROM scraped_data.washing_machine WHERE brand ILIKE \'%Bosch%\' AND max_load >= 5 and rating_value >= 4.8;'
                      f'\n\nUser input: хорошая стиральная машина\nSQL query: SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE rating_value >= 4.5;'
                      f'\n\nUser input: дешевая\nSQL query: SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE price <= 1000;'
                      f'\n\nUser input: Производители Атлант, Хаер, Ханса, Самсунг\nSQL query: SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE brand ILIKE ANY(ARRAY[\'%Atlant%\',\'%Haier%\',\'%Hansa%\',\'%Samsung%\']);'
                      f'\n\n\nUser input: {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n```sql')


        #api call from string
        response = call_generation_api(prompt=str_prompt, grammar=None, stop=['<|eot_id|>', '```', '```\n',])
        response_query = response.get('generation')

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
        if 'name' in df.columns:
            if df['name'].nunique() != df.shape[0]:
                if 'price' in df.columns:
                    df.sort_values(['name', 'price'], inplace=True)
                    df.drop_duplicates(subset=['name'], keep='first', inplace=True)
        logger.warning(f'{df.shape}')
        logger.warning(f'{df.head()}')
        return df


if __name__ == '__main__':
    # user_input = "хорошая стиральная машина глубина до 43, загрузка от 6"
    # user_input = "дешевая стиральная машина"
    # user_input = 'Поможете найти недорогую стиральную машину, которая работает хорошо?'
    user_input = 'Недорогая стиральная машина с хорошими характеристиками.'
    response = SqlToText().sql_query(user_input=user_input)
    print(response)


