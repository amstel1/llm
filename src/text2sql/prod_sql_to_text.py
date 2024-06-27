# from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
# from langchain.chains import create_sql_query_chain
# from langchain_community.llms import LlamaCpp
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
# from operator import itemgetter
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
from loguru import logger
# import os
# from langfuse import Langfuse
# from langfuse.callback import CallbackHandler
# from datetime import datetime
# from langchain_community.agent_toolkits import create_sql_agent
# import pandas as pd
import sys
sys.path.append('/home/amstel/llm/src')
from postgres.config import user, password, host, port, database
from typing import List
from general_llm.llm_endpoint import call_generate_from_query_api, call_generation_api, call_generate_from_history_api
from sqlalchemy.sql import text
from rag.rag_config import N_EMBEDDING_RESULTS, EMBEDDING_MODEL_NAME, ELBOW_EMBEDDING, MOST_RELEVANT_AT_THE_TOP
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
    MilvusClient
)
import json
from sqlparse.sql import Where
import sqlparse
from sqlparse import tokens as T


def update_sql_statement(sql_statement: str, new_where_clause: str):
    sql_statement = sql_statement.replace('  ', ' ').replace('\n', ' ').replace('\t', ' ')
    assert 'where' in new_where_clause.lower().split(' ')
    parsed = list(sqlparse.parse(sql_statement))[0]
    if 'where' in sql_statement.lower():
        # replace
        new_sql_statement = ' '.join((parsed_token.value.strip() if not isinstance(parsed_token, Where) else new_where_clause for parsed_token in parsed.tokens ))
    else:
        # add in the right place
        insert_index = len(parsed.tokens)  # Default: insert at the end
        for i, item in enumerate(parsed.tokens):
            if item.ttype is T.Keyword and item.value.upper() == 'FROM':
                insert_index = i
                break

        # Insert the new WHERE clause
        parsed.tokens.insert(insert_index+3, sqlparse.parse(new_where_clause)[0])
        new_sql_statement = ' '.join((parsed_token.value.strip() for parsed_token in parsed.tokens))
    return new_sql_statement


class SqlToText:
    # name, price, rating
    mandatory_fields = [
        ('name', 'text', 'название'),  # attribute_name_eng, attribute_type, attribute_rus
        ('price', 'real', 'цена'),
        ('rating_value', 'real', 'рейтинг'),
    ]
    mandatory_fields_eng = [x[0] for x in mandatory_fields]

    def create_str_description(self, fields: list):
        result = ""
        for field in fields:
            result += f'"{field[0]}" {field[1]}, -- {field[2]}\n'
        return result

    def create_table_description(self, schema_name: str, table_name: str, body: str):
        begin = f"Here is the relevant table info:\ncreate table {schema_name}.{table_name} (\n"
        end = ")\n\n"
        return f"{begin}{body}{end}"

    def create_few_shot_examples(self, qa_pairs: list):
        result = "Here are the examples of correct pairs of Q and SQL :\n"
        for pair in qa_pairs:
            q, a = pair
            result += f"Q: {q.strip()}\nSQL: {a.strip()}\n"
        return result

    def create_prompt(self, prefix, table_description, body, postfix):
        return prefix + table_description + body + postfix

    def postprocess_df(self, df):
        logger.info(f'sql df - before filters on price and rating -- {df.shape}')
        if 'price' in df.columns: df = df[df.price.notnull()]
        if 'rating_value' in df.columns: df = df[df.rating_value.notnull()]
        logger.info(f'sql df - after filters on price and rating -- {df.shape}')
        if 'name' in df.columns:
            if df['name'].nunique() != df.shape[0]:
                if 'rating_count' in df.columns:
                    df.sort_values(['rating_count',], ascending=[False,], inplace=True)
                df.drop_duplicates(subset=['name'], keep='first', inplace=True)
        logger.warning(f'{df.shape}')
        logger.warning(f'{df.head()}')
        return df

    def sql_query(self, schema_name: str, user_query: str, predefined_sql: str = None) -> pd.DataFrame | str:
        '''returns df with results and sql query'''
        assert schema_name in ('washing_machine', 'fridge', 'tv', 'mobile')
        uri = f"postgresql://{host}:{port}/{database}?user={user}&password={password}"
        if predefined_sql:
            # disregard user_query, just execute sql statement
            df = pd.read_sql(
                sql=predefined_sql,
                con=uri,
            )
            df = self.postprocess_df(df)
            return df, predefined_sql

        # index user_query
        dense_embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,  # Specify the model name
            model_kwargs={'device': 'cpu',}
        )
        # fields = [
        #     FieldSchema(name="attribute_name_eng", dtype=DataType.VARCHAR, max_length=1024),
        #     FieldSchema(name="attribute_name_rus", dtype=DataType.VARCHAR, is_primary=True, max_length=1024, ),
        #     FieldSchema(name="attribute_name_rus_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        #     FieldSchema(name="attribute_type", dtype=DataType.VARCHAR, max_length=4),  # may only be real / text1
        # ]
        # CONNECTION_URI = "http://localhost:19530"
        # connections.connect(uri=CONNECTION_URI, db_name=schema_name)

        data = dense_embedding_model.embed_query(text=user_query)

        # create table description
        client = MilvusClient(db_name=schema_name)
        attributes_retrieved = client.search(
            collection_name="postgres_table_attributes",  # Replace with the actual name of your collection
            # Replace with your query vector
            data=[data],
            limit=N_EMBEDDING_RESULTS,  # Max. number of search results to return
            search_params={"metric_type": "IP", "params": {}},  # Search parameters
            output_fields=['attribute_name_eng', 'attribute_name_rus', 'attribute_type']
        )

        json_attributes_retrieved = attributes_retrieved[0] #json.load(attributes_retrieved)
        assert isinstance(json_attributes_retrieved, list)
        fields = []
        for jsn in json_attributes_retrieved:
            entity = jsn.get('entity')
            assert isinstance(entity, dict)
            if entity.get("attribute_name_eng") not in SqlToText.mandatory_fields_eng:
                fields.append((entity.get("attribute_name_eng"), entity.get("attribute_type"), entity.get("attribute_name_rus"), ))
        body_mandatory = self.create_str_description(fields=self.mandatory_fields)
        body_nonmandatory = self.create_str_description(fields=fields)
        body = body_mandatory + body_nonmandatory
        table_description = self.create_table_description(schema_name=schema_name, table_name=schema_name, body=body)

        # get most relevant Q&A examples for the few-shots
        examples_retrieved = client.search(
            collection_name="q_a_index",  # Replace with the actual name of your collection
            # Replace with your query vector
            data=[data],
            limit=N_EMBEDDING_RESULTS,  # Max. number of search results to return
            search_params={"metric_type": "IP", "params": {}},  # Search parameters
            output_fields=['q', 'q_vector', 'a']
        )
        assert isinstance(examples_retrieved, list)
        json_examples_retrieved = examples_retrieved[0]
        assert isinstance(json_examples_retrieved, list)
        # json_examples_retrieved = json.load(examples_retrieved)
        examples = []
        for jsn in json_examples_retrieved:
            entity = jsn.get('entity')
            assert isinstance(entity, dict)
            examples.append((entity.get("q"), entity.get("a"),))
        few_shots = self.create_few_shot_examples(qa_pairs=examples)
        str_prompt = self.create_prompt(
            prefix='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n'\
'You are a top class business analyst that specializes in translating natural language queries into SQL. Perform the task you are assigned to to the best of your ability. <|eot_id|><|start_header_id|>user<|end_header_id|>\nGiven a Q, create a valid SQL to run. Access only the attributes present in the table definition.\n\n',
            table_description=table_description,
            body=few_shots,
            postfix=f"""\n\nUser query:\nQ: {user_query.strip()} 
SQL:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n```sql""",
        )

        # .replace('Суть требования пользователя: ', '').replace('Суть требований пользователя: ', '')
        # construct

        # trace_name = f'sql_llama3_{datetime.now()}'
        # os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
        # os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-8c20497b-23da-4267-961c-f66e33a8bee4"
        # os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-641f1b97-5f99-456a-8597-a44a8f7fc6ab"
        # langfuse = Langfuse()
        # langfuse_callback_handler = CallbackHandler(trace_name=trace_name)



        # table_info = '''create table washing_machine.washing_machine (
        #       "brand" text, -- название производителя
        #       "rating_value" real, -- рейтинг товара
        #       "rating_count" real, -- количество оценок
        #       "review_count" real, -- количество отзывов
        #       "name" text, -- название товара
        #       "price" real, -- цена, руб.
        #       "max_load" real, -- максимальная загрузка, кг.
        #       "depth" real, -- глубина, см.
        #       "drying" text -- есть ли сушка, Да/Нет
        #     )
        #
        #     /*
        #     3 rows from scraped_data.washing_machine table:
        #     brand|rating_value|rating_count|review_count|name|price|max_load|depth|drying
        #     Candy|4.5|966|299|Стиральная машина Candy AQUA 114D2-07|882.99|4|43|Нет
        #     LG|5|6|2|Стиральная машина LG F2J3WS2W|1273.91|6.5|44|Да
        #     Indesit|4.5|921|290|Стиральная машина Indesit IWSB 51051 BY|120|5|40|Нет
        #     */'''
        #
        # examples = [
        #     {"input": "Фирма Электролюкс без сушки глубина до 50 см",
        #      "query": "SELECT name, price, rating_value, drying, depth FROM scraped_data.washing_machine WHERE brand ILIKE '%Electrolux%' AND (drying = 'Нет' or drying is null) AND depth <= 50;"},
        #     {"input": "популярная, хорошая, недорогая",
        #      "query": "SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE price <= 1000;"},
        #     {"input": "Cтиральная машина с сушкой",
        #      "query": "SELECT name, price, rating_value, drying FROM scraped_data.washing_machine WHERE drying = 'Да';"},
        #     {"input": "отличная стиралка",
        #      "query": "SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE rating_value >= 4.8;"},
        #     {"input": "Бош, от 5 кг, хорошая лучшая",
        #      "query": "SELECT name, price, rating_value, max_load FROM scraped_data.washing_machine WHERE brand ILIKE '%Bosch%' AND max_load >= 5 and rating_value >= 4.8;"},
        #     {"input": "хорошая стиральная машина",
        #      "query": "SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE rating_value >= 4.5;"},
        #     {"input": "дешевая",
        #      "query": "SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE price <= 1000;"},
        #     {"input": "Производители Атлант, Хаер, Ханса, Самсунг",
        #      "query": "SELECT name, price, rating_value FROM scraped_data.washing_machine WHERE brand ILIKE ANY(ARRAY['%Atlant%','%Haier%','%Hansa%','%Samsung%']);"},
        # ]
        #
        # example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")
        #
        # system_message_llama = (
        #     '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n'
        #     'You are a postgresql expert. Given an input question, create a syntactically correct postgresql query to run. ' \
        #     ' Here are the requirements you must follow:\n'
        #     ' translate "brand" into latin characters;\n'
        #     ' if needed to filter by fields "brand", "name" you MUST use operator "ILIKE" with "%" instead of "=";\n'
        #     ' filter by field "brand" only if "brand" is mentioned in "User input";\n\n'
        #     'Here is the relevant table info: \n{table_info}\n\n'
        # )

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
#         str_prompt = (
# '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n'\
# 'You are a top class business analyst that specializes in translating natural language queries into SQL. Perform the task you are assigned to to the best of your ability. <|eot_id|><|start_header_id|>user<|end_header_id|>\n'\
# f"""Given an input_query, create a valid SQL query to run.
#
#
# Here is the relevant table info:
# create table scraped_data.washing_machine (
#  "brand" text, -- название производителя
# "rating_value" real, -- рейтинг товара
# "rating_count" real, -- количество оценок
# "review_count" real, -- количество отзывов
# "name" text, -- название товара
# "price" real, -- цена, руб.
# "max_load" real, -- максимальная загрузка, кг.
# "depth" real, -- глубина, см.
# "drying" text -- есть ли сушка, Да/Нет
# )
#
#
# Here are the rows from scraped_data.washing_machine table:
# brand | rating_value | rating_count | review_count | name | price | max_load | depth | drying
# Candy | 4.5 | 966 | 299 | Стиральная машина Candy AQUA 114D2-07 | 882.99 | 4 | 43 | Нет
# LG | 5 | 6 | 2 |Стиральная машина LG F2J3WS2W | 1273.91 | 6.5 | 44 | Да
# Indesit | 4.5 | 921 | 290 | Стиральная машина Indesit IWSB 51051 BY | 120 | 5 | 40 |Нет
#
#
# Here are the examples of correct pairs of input_query (Q) and required output (SQL):
#
# Q: Фирма Электролюкс без сушки глубина до 50 см;
# SQL: SELECT name, price, rating_value, drying, depth FROM washing_machine.washing_machine WHERE (brand ILIKE '%Electrolux%') AND (drying = 'Нет' or drying is null) AND depth <= 50;
#
# Q: популярная, хорошая, недорогая
# SQL: SELECT name, price, rating_value FROM washing_machine.washing_machine WHERE price <= 1000;
#
# Q: Cтиральная машина с сушкой
# SQL: SELECT name, price, rating_value, drying FROM washing_machine.washing_machine WHERE drying = 'Да';
#
# Q: отличная стиралка;
# SQL: SELECT name, price, rating_value FROM washing_machine.washing_machine WHERE rating_value >= 4.8;
#
# Q: от 5 кг, хорошая лучшая
# SQL: SELECT name, price, rating_value, max_load FROM washing_machine.washing_machine WHERE (max_load >= 5) and (rating_value >= 4.8);
#
# Q: хорошая стиральная машина
# SQL: SELECT name, price, rating_value FROM washing_machine.washing_machine WHERE rating_value >= 4.5;
#
# Q: дешевая
# SQL: SELECT name, price, rating_value FROM washing_machine.washing_machine WHERE price <= 1000;
#
#
# input_query: {user_query}
# SQL:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n```sql""")

        logger.warning('str_prompt')
        logger.warning(str_prompt)
        #api call from string
        response = call_generation_api(prompt=str_prompt, grammar=None, stop=['<|eot_id|>', '```', '```\n',])
        response_query = re.sub(r'(?i)\blike\b', 'ilike', response.strip().replace('\n', ' '))
        response_query = text(response_query)
        # replace like with ilike, but not ilike

        logger.warning(response_query)


        df = pd.read_sql(
            sql=response_query,
            con=uri,
        )
        df = self.postprocess_df(df)
        sql_query = response_query.text
        return df, sql_query


if __name__ == '__main__':
    # user_query = "стиральная машина глубина до 43, загрузка от 6, с рейтингом от 4.8, производитель Korting"
    # user_query = "дешевая стиральная машина"
    # user_query = 'Поможете найти недорогую стиральную машину, которая работает хорошо?'
    # user_query = 'Недорогая стиральная машина с хорошими характеристиками.'
    user_query = "Суть запроса: холодильник до 2500 руб, фирма lg, высота от 195"
    # user_query = "Суть требований пользователя: стиральная машина с хорошим брендом, узкой и вместительной."
    response = SqlToText().sql_query(schema_name='fridge', user_query=user_query)
    print(response)


