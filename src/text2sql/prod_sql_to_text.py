import sql_metadata
from loguru import logger
import sys
import pickle
# from scripts.trash.chatbot_quickstart import retriever

sys.path.append('/home/amstel/llm/src')
from postgres.config import user, password, host, port, database
from typing import List
from general_llm.llm_endpoint import call_generate_from_query_api, call_generation_api, call_generate_from_history_api, MODEL_NAME
from sqlalchemy.sql import text

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
    MilvusClient,
    RRFRanker,
)
import json
from sqlparse.sql import Where
import sqlparse
from sqlparse import tokens as T
from general_llm.prompt_construction import Llama3PromptTemplate, Gemma2PromptTemplate, ChatMLPromptTemplate, Qwen25PromptTemplate
import sqlparse
from sqlparse.sql import Where, Identifier, Comparison
from sqlparse.tokens import Keyword, DML
import sql_metadata

from rag.utils import ExtendedMilvusCollectionHybridSearchRetriever as MilvusCollectionHybridSearchRetriever
from rag.utils import BGEDocumentCompressor
from rag.rag_config import N_RERANK_RESULTS, USE_RERANKER, RERANKING_MODEL, ELBOW_RERANKING
from langchain.retrievers import ContextualCompressionRetriever
from rag.rag_config import N_EMBEDDING_RESULTS, EMBEDDING_MODEL_NAME, ELBOW_EMBEDDING, MOST_RELEVANT_AT_THE_TOP


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

def update_sql_statement_append_where_condition(sql_statement: str, new_where_clause: str):
    sql_statement = sql_statement.replace('  ', ' ').replace('\n', ' ').replace('\t', ' ')
    assert 'where' in new_where_clause.lower().split(' ')
    parsed = list(sqlparse.parse(sql_statement))[0]
    if 'where' in sql_statement.lower():
        # replace
        new_sql_statement = ' '.join((parsed_token.value.strip()
                                      if not isinstance(parsed_token, Where)
                                      else parsed_token.value.replace(';','') + ' ' + new_where_clause.lower().replace('where', 'and')
                                      for parsed_token in parsed.tokens))
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
    if not new_sql_statement.strip().endswith(';'):
        new_sql_statement = new_sql_statement.strip()+';'
    assert 'insert' not in new_sql_statement
    assert 'update' not in new_sql_statement
    return new_sql_statement

def extract_where_attributes(sql_query: str)  -> list[str]:
    """
    :param sql_query: valid sql query
    :return: list of str attributes from WHERE clause
    """
    # Parse the SQL statement
    post_where_list = [x.lower() for x in sql_query[sql_query.lower().find('where'):].split(" ")]
    attributes = sql_metadata.Parser(sql_query).columns_dict.get('where', [])
    # do the same for each keyword that matches an sql column name
    if 'type' in post_where_list and 'type' not in attributes:
        logger.warning('sql query, where parsing - appending "type"')
        attributes.append('type')
    return attributes

class SqlToText:
    # attribute_name_eng, attribute_type, attribute_rus
    mandatory_fields = [
        ('name', 'text', 'название'),
        ('price', 'real', 'цена'),
        ('rating_value', 'real', 'рейтинг'),
        ('rating_count', 'real', 'количество рейтингов (оценок)'),
        ('review_count', 'real', 'количество отзывов'),
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
        # if 'rating_value' in df.columns: df = df[df.rating_value.notnull()]  # temporarily disable this while not all phones are scraped
        logger.info(f'sql df - after filters on price and rating -- {df.shape}')
        if 'name' in df.columns:
            if df['name'].nunique() != df.shape[0]:
                # if 'rating_count' in df.columns:
                #     df.sort_values(['rating_count',], ascending=[False,], inplace=True)
                df.drop_duplicates(subset=['name'], keep='first', inplace=True)
        logger.warning(f'{df.shape}')
        logger.warning(f'{df.head()}')
        return df

    def _get_retiever(self, schema_name:str, user_query:str):
        user_query = user_query.lower()  # need this decide if we can use nybrid instead of dense retriever

        sparse_search_params = {"metric_type": "IP"}
        dense_search_params = {"metric_type": "IP", "params": {}}

        CONNECTION_URI = "http://localhost:19530"
        connections.connect(uri=CONNECTION_URI, db_name=schema_name)
        # these are fields in milvus Q(user says in russian)&A(sql query) few shot prompt
        fields = [
            FieldSchema(name="q", dtype=DataType.VARCHAR, is_primary=True, max_length=2048, ),
            FieldSchema(name="q_dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="q_sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="a", dtype=DataType.VARCHAR, max_length=2048),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)  # зачем?
        collection = Collection(name='q_a_index', schema=schema)  # schema Schema type must be schema.CollectionSchema
        if not self.sparse_embedding_model.embed_query(user_query):
            # empty bm25
            retriever = MilvusCollectionHybridSearchRetriever(
                collection=collection,
                rerank=WeightedRanker(1.0),
                anns_fields=['q_dense_vector'],
                field_embeddings=[self.dense_embedding_model],
                field_search_params=[dense_search_params],
                top_k=N_EMBEDDING_RESULTS,
                text_field="a",
                use_elbow_for_embedding=ELBOW_EMBEDDING,
            )
        else:
            retriever = MilvusCollectionHybridSearchRetriever(
                collection=collection,
                rerank=WeightedRanker(0.75, 0.25),
                # rerank=RRFRanker(k=60),
                anns_fields=['q_dense_vector', 'q_sparse_vector'],
                field_embeddings=[self.dense_embedding_model, self.sparse_embedding_model],
                field_search_params=[dense_search_params, sparse_search_params],
                top_k=N_EMBEDDING_RESULTS,
                text_field="a",
                use_elbow_for_embedding=ELBOW_EMBEDDING,
            )


        if USE_RERANKER:
            compressor = BGEDocumentCompressor(
                top_n=N_RERANK_RESULTS,
                model_name_or_path=RERANKING_MODEL,
                elbow=ELBOW_RERANKING,
                most_relevant_at_the_top=MOST_RELEVANT_AT_THE_TOP,
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            return compression_retriever
        return retriever

    def sql_query(self, schema_name: str, user_query: str, predefined_sql: str = None) -> pd.DataFrame | str:
        '''returns df with results and sql query'''

        # load dense and sparse retrieval models for this schema
        self.dense_embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,  #
            model_kwargs={'device': 'cpu',}
        )
        with open(f'/home/amstel/llm/src/rag/FewShotQA_{schema_name}.pkl', 'rb') as f:
            self.sparse_embedding_model = pickle.load(f)
        assert self.dense_embedding_model
        assert self.sparse_embedding_model

        user_query = user_query.lower()
        assert schema_name in ('washing_machine', 'fridge', 'tv', 'mobile',)
        uri = f"postgresql://{host}:{port}/{database}?user={user}&password={password}"
        if predefined_sql:
            # disregard user_query, just execute sql statement
            logger.warning(predefined_sql)
            df = pd.read_sql(
                    sql=predefined_sql,
                con=uri,
            )
            df = self.postprocess_df(df)
            return df, predefined_sql



        retriever = self._get_retiever(schema_name=schema_name, user_query=user_query)
        json_examples_retrieved = retriever.invoke(user_query)
        assert isinstance(json_examples_retrieved, list)
        # json_examples_retrieved = json.load(examples_retrieved)
        examples = []
        answers = []
        for jsn in json_examples_retrieved:
            q = jsn.metadata.get("q")
            a = jsn.page_content
            examples.append((q, a))
            answers.append(jsn.page_content)
        few_shots = self.create_few_shot_examples(qa_pairs=examples)

        where_attributes_few_shots = set()
        for answer in answers:
            extracted_where_attributes = extract_where_attributes(answer)
            logger.critical(f'1211 - answer: {answer}')
            logger.critical(f'1211 - extracted_where_attributes: {extracted_where_attributes}')
            where_attributes_few_shots.update(extracted_where_attributes)
        ###################################### start
        # create table description

        # 21 11 2024 - нет необходимости подтягивать атрибуты для описания таблицы гибридом, dense достаточно
        # через sqlparse смотрим все атрибуты where из few-shot и тянем их сюда

        data = self.dense_embedding_model.embed_query(text=user_query.lower())
        client = MilvusClient(db_name=schema_name)

        attributes_retrieved = client.search(
            collection_name="postgres_table_attributes",  # list of attrs for each
            data=[data],
            limit=1000,  # retrieve all
            search_params={"metric_type": "IP", "params": {}},  # Search parameters
            output_fields=['attribute_name_eng', 'attribute_name_rus', 'attribute_type']
        )
        logger.error(f'0811 - attributes_retrieved: {attributes_retrieved}')
        json_attributes_retrieved = attributes_retrieved[0] #json.load(attributes_retrieved)
        assert isinstance(json_attributes_retrieved, list)
        fields = []
        for ix, jsn in enumerate(json_attributes_retrieved):
            # ix is sequence number of items sorted by relevance
            entity = jsn.get('entity')
            assert isinstance(entity, dict)
            if entity.get("attribute_name_eng") in SqlToText.mandatory_fields_eng:
                continue
            elif entity.get("attribute_name_eng") in where_attributes_few_shots and \
               entity.get("attribute_name_eng") not in SqlToText.mandatory_fields_eng:
                fields.append((entity.get("attribute_name_eng"), entity.get("attribute_type"), entity.get("attribute_name_rus"), ))
            elif entity.get("attribute_name_eng") not in where_attributes_few_shots and \
               entity.get("attribute_name_eng") not in SqlToText.mandatory_fields_eng and \
               ix <= 3:
                fields.append((entity.get("attribute_name_eng"), entity.get("attribute_type"), entity.get("attribute_name_rus"), ))
        body_mandatory = self.create_str_description(fields=self.mandatory_fields)
        body_nonmandatory = self.create_str_description(fields=fields)
        body = body_mandatory + body_nonmandatory
        table_description = self.create_table_description(schema_name=schema_name, table_name=schema_name, body=body)
        ###################################### end



        system_prompt = 'You are a top class business analyst that specializes in translating natural language queries into SQL. Perform the task you are assigned to to the best of your ability.'
        user_prompt_pt1 = '\nGiven a Q, create a valid SQL to run. Access only the attributes present in the table definition.\n\n'
        user_prompt_pt2 = table_description
        user_prompt_pt3 = few_shots
        user_prompt_pt4 = f"\n\nUser query:\nQ: {user_query.strip()}\nSQL:"
        user_prompt = user_prompt_pt1 + user_prompt_pt2 + user_prompt_pt3 + user_prompt_pt4
        assistant_must_start_with = '\n```sql'
        if 'llama' in MODEL_NAME: prompt_template = Llama3PromptTemplate
        if 'gemma' in MODEL_NAME: prompt_template = Gemma2PromptTemplate
        if 'chatml' in MODEL_NAME: prompt_template = ChatMLPromptTemplate
        if 'qwen' in MODEL_NAME: prompt_template = Qwen25PromptTemplate
        str_prompt = prompt_template().create_prompt_from_user_query(
            system_prompt_clean=system_prompt,
            user_query=user_prompt,
            assistant_must_start_with=assistant_must_start_with,
        )

        logger.warning('str_prompt')
        logger.warning(str_prompt)
        #api call from string
        response = call_generation_api(prompt=str_prompt, grammar=None, stop=['<|eot_id|>', '```', '```\n',])
        response_query = re.sub(r'(?i)\blike\b', 'ilike', response.strip().replace('\n', ' '))
        response_query = text(response_query)
        response_query = 'SELECT * ' + response_query[response_query.upper().find('FROM'):]  # user might ask for specific attribute, we get all
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
    response = SqlToText().sql_query(schema_name='fridge', user_query=user_query,
                                     # predefined_sql="SELECT * FROM washing_machine.washing_machine WHERE name ILIKE '%%Electrolux%%' and price <= 3000;"
                                     )
    print(response)


